#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マウス Geneformer in silico perturbation 統計テンプレート
==========================================================

`InSilicoPerturberStats` を CLI 化。deletion の結果（cosine shift）から
goal-state shift を集計する。null 分布を渡すと p 値 / FDR も算出される。

注意:
  * 本リポジトリ移動により `GENE_NAME_ID_DICTIONARY_FILE` 既定値が壊れているため、
    `--gene-name-id-dict` を自動解決して明示的に渡す。
  * `get_stats(input_data_directory, null_dist_data_directory, output_directory,
    output_prefix)` は **null を位置引数で要求**する（不要なら None）。
  * 入力ディレクトリに `*_raw.pickle` が 1 つも無いと例外になる
    （失敗した遺伝子の空ディレクトリはスキップすること）。

使用例:
  python template_isp_stats.py \
      --input-dir /path/to/outputs/isp/leiden_start6_goal0_delete_Olfm4 \
      --output-dir /path/to/outputs/isp_stats \
      --prefix leiden_start6_goal0_delete_Olfm4 \
      --label leiden --start 6 --goal 0

  # ディレクトリ配下の遺伝子別 run をまとめて処理（--each-subdir）
  python template_isp_stats.py --each-subdir --input-dir /path/to/outputs/isp \
      --glob "leiden_start6_goal0_delete_*" \
      --output-dir /path/to/outputs/isp_stats --prefix leiden_start6_goal0 \
      --label leiden --start 6 --goal 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GENECORPUS = REPO / "data" / "Mouse-Genecorpus-20M"
DEFAULT_TOKEN_DICT = GENECORPUS / "MLM-re_token_dictionary_v1.pkl"
DEFAULT_GENE_NAME_ID = GENECORPUS / "MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"


def main() -> int:
    ap = argparse.ArgumentParser(description="Mouse-Geneformer ISP statistics")
    ap.add_argument("--input-dir", required=True, help="ISP 出力ディレクトリ")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--mode", default="goal_state_shift",
                    choices=["goal_state_shift", "vs_null", "mixture_model", "aggregate_data"])
    ap.add_argument("--label", default=None, help="状態ラベルの列名（goal_state_shift では必須）")
    ap.add_argument("--start", default=None)
    ap.add_argument("--goal", default=None)
    ap.add_argument("--alt", nargs="*", default=None)
    ap.add_argument("--genes-perturbed", default="all")
    ap.add_argument("--null-dir", default=None,
                    help="null 分布ディレクトリ（渡すと p 値 / FDR が出る）")
    ap.add_argument("--each-subdir", action="store_true",
                    help="input-dir 直下の各サブディレクトリを個別に処理")
    ap.add_argument("--glob", default="*",
                    help="--each-subdir 時のサブディレクトリ選択パターン")
    ap.add_argument("--token-dict", default=str(DEFAULT_TOKEN_DICT))
    ap.add_argument("--gene-name-id-dict", default=str(DEFAULT_GENE_NAME_ID))
    args = ap.parse_args()

    from geneformer import InSilicoPerturberStats

    states = None
    if args.mode == "goal_state_shift":
        if not (args.label and args.start and args.goal):
            sys.exit("goal_state_shift requires --label/--start/--goal")
        states = {"state_key": args.label, "start_state": args.start,
                  "goal_state": args.goal, "alt_states": args.alt or []}

    def run_one(in_dir: Path, out_prefix: str, null_dir) -> bool:
        if not list(in_dir.glob("*_raw.pickle")):
            print(f"  skip (no *_raw.pickle): {in_dir}", flush=True)
            return False
        print(f"  stats: {out_prefix}", flush=True)
        st = InSilicoPerturberStats(
            mode=args.mode,
            genes_perturbed=args.genes_perturbed,
            combos=0, anchor_gene=None,
            cell_states_to_model=states,
            token_dictionary_file=args.token_dict,
            gene_name_id_dictionary_file=args.gene_name_id_dict,
        )
        st.get_stats(
            input_data_directory=str(in_dir),
            null_dist_data_directory=null_dir,
            output_directory=args.output_dir,
            output_prefix=out_prefix,
        )
        return True

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.each_subdir:
        subs = sorted(d for d in Path(args.input_dir).glob(args.glob) if d.is_dir())
        print(f"{len(subs)} subdirs", flush=True)
        n = sum(run_one(d, d.name, args.null_dir) for d in subs)
        print(f"processed {n}/{len(subs)}", flush=True)
    else:
        run_one(Path(args.input_dir), args.prefix, args.null_dir)
    print(f"done -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
