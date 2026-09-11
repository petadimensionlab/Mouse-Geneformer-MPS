#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マウス Geneformer トークン化テンプレート
=======================================

notebook / `execute_tokenizer.py` の手順を、再利用できる CLI スクリプトにしたもの。

元コードとの違い・注意:
  * 本リポジトリは `~/workspace/Mouse-Geneformer-MPS` から
    `~/workspace/research/Mouse-Geneformer-MPS` へ移動しており、
    `geneformer/tokenizer.py` 内の `GENE_MEDIAN_FILE` /
    `TOKEN_DICTIONARY_FILE` は**旧パスで壊れている**。
    本テンプレートはリポジトリ位置から実パスを自動解決して明示的に渡す
    （リポジトリ本体は書き換えない）。
  * 入力の探索は `Path(data_directory).glob("*.h5ad")` で**ディレクトリ内の全 h5ad**を
    処理し、1 ファイルにつき `<prefix>_<番号>.dataset` を出力する。
    → **入力 h5ad は専用ディレクトリに 1 ファイルだけ置くこと。**
  * h5ad の要件: `var["ensembl_id"]`（マウスは ENSMUSG）、`obs["n_counts"]`。
    ラベルとして運びたい obs 列は `--attrs` で指定。

使用例:
  python template_tokenize.py \
      --input-dir  /path/to/mouse_input/ \
      --output-dir /path/to/outputs/tokenized \
      --prefix GSE273983_mouse \
      --attrs leiden region_cluster_corr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- リポジトリ内の既定パスを自動解決（壊れたハードコードの回避）---------------
REPO = Path(__file__).resolve().parents[1]
GENECORPUS = REPO / "data" / "Mouse-Genecorpus-20M"
DEFAULT_GENE_MEDIAN = GENECORPUS / "mouse_gene_median_dictionary.pkl"
DEFAULT_TOKEN_DICT = GENECORPUS / "MLM-re_token_dictionary_v1.pkl"


def _make_pandas2_safe_tokenize_anndata():
    """pandas 2.x / anndata 0.13 でも動く tokenize_anndata を作って返す。

    元実装からの修正点:
      * `ad.read(...)`（anndata 0.11 で削除）→ `ad.read_h5ad(...)`
      * `adata.var["ensembl_id"][loc]` は pandas 2.x では **ラベル参照**になり
        KeyError になる → `np.asarray(...)[loc]` に変更
      * それ以外（正規化・rank encoding・カスタム属性の持ち越し）は元と同じ
    """
    def _tokenize_anndata(self, adata_file_path, target_sum=10_000, chunk_size=512):
        import anndata as ad
        import numpy as np
        import scipy.sparse as sp
        from geneformer.tokenizer import rank_genes

        adata = ad.read_h5ad(adata_file_path, backed="r")

        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        ens = np.asarray(adata.var["ensembl_id"])
        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in ens]
        )[0]
        norm_factor_vector = np.array(
            [self.gene_median_dict[ens[i]] for i in coding_miRNA_loc]
        )
        coding_miRNA_ids = ens[coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )
        print(f"tokenizing {coding_miRNA_loc.shape[0]} genes "
              f"x {adata.shape[0]} cells", flush=True)

        try:
            _ = adata.obs["filter_pass"]
            var_exists = True
        except KeyError:
            var_exists = False

        if var_exists:
            filter_pass_loc = np.where([i == 1 for i in adata.obs["filter_pass"]])[0]
        else:
            print(f"{adata_file_path} has no 'filter_pass'; tokenizing all cells.",
                  flush=True)
            filter_pass_loc = np.array([i for i in range(adata.shape[0])])

        tokenized_cells = []
        for i in range(0, len(filter_pass_loc), chunk_size):
            idx = filter_pass_loc[i:i + chunk_size]
            n_counts = adata[idx].obs["n_counts"].values[:, None]
            X_view = adata[idx, coding_miRNA_loc].X
            X_norm = (X_view / n_counts * target_sum / norm_factor_vector)
            X_norm = sp.csr_matrix(X_norm)
            tokenized_cells += [
                rank_genes(X_norm[j].data, coding_miRNA_tokens[X_norm[j].indices])
                for j in range(X_norm.shape[0])
            ]
            if self.custom_attr_name_dict is not None:
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += adata[idx].obs[k].tolist()
            else:
                file_cell_metadata = None

        return tokenized_cells, file_cell_metadata, len(filter_pass_loc)

    return _tokenize_anndata


def main() -> int:
    ap = argparse.ArgumentParser(description="Tokenize mouse scRNA-seq for mouse-Geneformer")
    ap.add_argument("--input-dir", required=True,
                    help="h5ad を 1 つだけ含むディレクトリ（glob(*.h5ad) で全件処理される）")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--attrs", nargs="*", default=[],
                    help="トークン化データへ運ぶ obs 列（列名 = custom attr 名）")
    ap.add_argument("--gene-median", default=str(DEFAULT_GENE_MEDIAN))
    ap.add_argument("--token-dict", default=str(DEFAULT_TOKEN_DICT))
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--file-format", default="h5ad", choices=["h5ad", "loom"])
    args = ap.parse_args()

    for p in (args.gene_median, args.token_dict):
        if not Path(p).exists():
            sys.exit(f"required file not found: {p}")

    # --- 互換シム -----------------------------------------------------------
    # anndata >= 0.11 で `anndata.read` が削除されたが、マウス版 tokenizer は
    # `ad.read(...)` を呼ぶため、そのままだと AttributeError で落ちる。
    # パッケージ本体は書き換えず、実行時に別名を生やして回避する。
    import anndata as _ad
    if not hasattr(_ad, "read"):
        _ad.read = _ad.read_h5ad

    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob(f"*.{args.file_format}"))
    if not files:
        sys.exit(f"no *.{args.file_format} in {in_dir}")
    if len(files) > 1:
        print(f"WARNING: {len(files)} files found; ALL will be tokenized "
              f"({[f.name for f in files]})", file=sys.stderr)

    from geneformer import TranscriptomeTokenizer

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={a: a for a in args.attrs},
        nproc=args.nproc,
        gene_median_file=args.gene_median,
        token_dictionary_file=args.token_dict,
    )
    print(f"tokenizing {[f.name for f in files]} -> {args.output_dir}", flush=True)

    # --- h5ad 経路のバグ回避 ------------------------------------------------
    # `gene_keys` / `genelist_dict` は tokenize_loom() の中でしか設定されず、
    # tokenize_anndata() はそれを参照するため、h5ad 入力だと AttributeError になる。
    # loom 経路と同じ内容（gene_median_dict のキー＝マウス遺伝子全体）を補う。
    if not hasattr(tk, "genelist_dict"):
        tk.gene_keys = list(tk.gene_median_dict.keys())
        tk.genelist_dict = dict(zip(tk.gene_keys, [True] * len(tk.gene_keys)))
        print(f"shim: genelist_dict = {len(tk.genelist_dict)} genes "
              f"(from gene_median_dict)", file=sys.stderr)

    # --- pandas 2.x / anndata 0.13 互換 ------------------------------------
    # リポジトリ側で修正済みなら何もしない（古い実装のときだけ差し替える）
    import inspect
    import types
    src = inspect.getsource(tk.tokenize_anndata)
    if "ad.read(" in src:
        tk.tokenize_anndata = types.MethodType(
            _make_pandas2_safe_tokenize_anndata(), tk)
        print("shim: tokenize_anndata -> pandas2/anndata0.13-safe version",
              file=sys.stderr)
    else:
        print("tokenize_anndata: repository version is already compatible",
              file=sys.stderr)

    tk.tokenize_data(
        data_directory=str(in_dir) + "/",   # 末尾スラッシュ必須
        output_directory=args.output_dir,
        output_prefix=args.prefix,
        file_format=args.file_format,
    )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
