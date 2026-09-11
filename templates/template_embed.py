#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マウス Geneformer 埋め込み抽出テンプレート（frozen / fine-tuned 共通）
=====================================================================

V1 系モデル（vocab = ENSMUSG、`<cls>` トークンなし）なので `--emb-mode cell` を使う。
（`cls` は存在しない。`cell` = 非パディングトークンの平均プール。）

注意:
  * `EmbExtractor` の `max_ncells` 既定は **1000**。全細胞を使うには
    `--max-ncells 0`（= None 扱い）を指定する。
  * `token_dictionary_file` は旧パスで壊れているため本テンプレートが自動解決して渡す。
  * `extract_embs` の戻り値はバージョンにより `DataFrame` か `(DataFrame, tensor)`。
    本テンプレートは両方に対応する。

使用例:
  python template_embed.py \
      --model-dir   /path/to/mouse-Geneformer-L12-E20 \
      --input-dataset /path/to/tokenized/GSE273983_mouse_0.dataset \
      --output-dir  /path/to/outputs/embeddings \
      --prefix pretrained_cell_embeddings \
      --labels leiden region_cluster_corr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_DICT = REPO / "data" / "Mouse-Genecorpus-20M" / "MLM-re_token_dictionary_v1.pkl"


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract mouse-Geneformer cell embeddings")
    ap.add_argument("--model-dir", required=True,
                    help="Pretrained なら事前学習重み、CellClassifier なら fine-tune 済み重み")
    ap.add_argument("--input-dataset", required=True, help="*.dataset（tokenize 出力）")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--model-type", default="Pretrained",
                    choices=["Pretrained", "GeneClassifier", "CellClassifier"])
    ap.add_argument("--num-classes", type=int, default=0,
                    help="CellClassifier のとき必須（分類クラス数）")
    ap.add_argument("--emb-mode", default="cell", choices=["cell", "gene"])
    ap.add_argument("--emb-layer", type=int, default=-1, choices=[-1, 0])
    ap.add_argument("--labels", nargs="*", default=None,
                    help="出力 CSV に同梱する dataset 列（例: leiden）")
    ap.add_argument("--labels-to-plot", nargs="*", default=None)
    ap.add_argument("--max-ncells", type=int, default=0,
                    help="0 = 全細胞（None）。既定の 1000 を避ける")
    ap.add_argument("--summary-stat", default=None, choices=[None, "mean", "median"])
    ap.add_argument("--forward-batch-size", type=int, default=100)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--token-dict", default=str(DEFAULT_TOKEN_DICT))
    args = ap.parse_args()

    if args.model_type == "CellClassifier" and not args.num_classes:
        sys.exit("--num-classes is required for CellClassifier")

    import pandas as pd
    from geneformer import EmbExtractor

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ex = EmbExtractor(
        model_type=args.model_type,
        num_classes=args.num_classes,
        emb_mode=args.emb_mode,
        cell_emb_style="mean_pool",
        filter_data=None,
        max_ncells=(None if args.max_ncells in (0, -1) else args.max_ncells),
        emb_layer=args.emb_layer,
        emb_label=args.labels,
        labels_to_plot=args.labels_to_plot,
        forward_batch_size=args.forward_batch_size,
        nproc=args.nproc,
        summary_stat=args.summary_stat,
        token_dictionary_file=args.token_dict,
    )
    print(f"extracting embeddings ({args.model_type}, {args.emb_mode}) ...", flush=True)
    res = ex.extract_embs(
        model_directory=args.model_dir,
        input_data_file=args.input_dataset,
        output_directory=args.output_dir,
        output_prefix=args.prefix,
    )
    df = res[0] if isinstance(res, tuple) else res
    if isinstance(df, pd.DataFrame):
        print("embeddings:", df.shape, flush=True)
    else:
        print("result type:", type(df), flush=True)
    print(f"done -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
