#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マウス Geneformer in silico perturbation（deletion）テンプレート
================================================================

`in_silico_perturbation.ipynb` の流れを CLI 化したもの。
V1 系モデルなので `emb_mode="cell"`（`<cls>` トークンは存在しない）。

重要な仕様:
  * **`genes_to_perturb` にリストを渡すと「そのセットを同時に削除」する群摂動**になり、
    事前フィルタで『リスト内の全遺伝子を含む細胞』が要求される
    （`"No cells in dataset contain all genes to perturb as a group."`）。
    → **個別 KO は 1 遺伝子 = 1 ラン**。本テンプレートは
    `--genes-file` の各遺伝子について ISP を回し、遺伝子ごとに出力ディレクトリを作る。
  * 遺伝子は **token dictionary のキー = ENSMUSG（マウス Ensembl ID）** で指定する。
  * goal-state shift 用の状態埋め込みは **ISP 内部で計算される**（上位版の
    `state_embs_dict` は不要）。
  * 本リポジトリ移動によりモジュール既定のパスが壊れているため、
    `token_dictionary_file` は自動解決して明示的に渡す。

genes-file の形式（JSON）:
  {"Olfm4": {"ensembl": "ENSMUSG00000022026", "role": "..."} , ...}
  もしくは単純なリスト ["ENSMUSG...", ...] でも可。

使用例:
  python template_isp_delete.py \
      --model-dir /path/to/outputs/runs/cellClassifier_leiden \
      --model-type CellClassifier --num-classes 9 \
      --input-dataset /path/to/tokenized/GSE273983_mouse_0.dataset \
      --output-dir /path/to/outputs/isp \
      --genes-file /path/to/genes_mouse.json \
      --label leiden --start 6 --goal 0
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_DICT = REPO / "data" / "Mouse-Genecorpus-20M" / "MLM-re_token_dictionary_v1.pkl"


def load_genes(path: str) -> list[tuple[str, str]]:
    """[(label, ensembl), ...] を返す。"""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [(g, g) for g in obj]
    out = []
    for sym, val in obj.items():
        ens = val.get("ensembl") if isinstance(val, dict) else val
        out.append((sym, ens))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Mouse-Geneformer in silico deletion (per gene)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-type", default="CellClassifier",
                    choices=["Pretrained", "GeneClassifier", "CellClassifier"])
    ap.add_argument("--num-classes", type=int, default=0)
    ap.add_argument("--input-dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--genes-file", required=True, help="JSON: {symbol: {ensembl:...}} or [...]")
    ap.add_argument("--label", required=True, help="状態ラベルの列名（例: leiden）")
    ap.add_argument("--start", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--alt", nargs="*", default=None,
                    help="省略時は dataset 内の他クラスを自動で使う")
    ap.add_argument("--perturb-type", default="delete",
                    choices=["delete", "overexpress", "inhibit", "activate"])
    ap.add_argument("--emb-mode", default="cell", choices=["cell", "cell_and_gene"])
    ap.add_argument("--emb-layer", type=int, default=-1, choices=[-1, 0])
    ap.add_argument("--forward-batch-size", type=int, default=32)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--max-ncells", type=int, default=0, help="0 = 無制限")
    ap.add_argument("--state-embs-file", dest="state_embs_file", default=None,
                    help="事前計算した state 埋め込みの .pt（無い場合は毎ラン内部計算）")
    ap.add_argument("--token-dict", default=str(DEFAULT_TOKEN_DICT))
    args = ap.parse_args()

    from datasets import load_from_disk
    from geneformer import InSilicoPerturber

    ds = load_from_disk(args.input_dataset)
    if args.label not in ds.features:
        sys.exit(f"state column '{args.label}' not in {list(ds.features)}")
    classes = sorted({str(v) for v in ds[args.label]})
    if args.start not in classes:
        sys.exit(f"start '{args.start}' not in {classes}")
    alt = args.alt if args.alt else [c for c in classes if c not in (args.start, args.goal)]
    states = {"state_key": args.label, "start_state": args.start,
              "goal_state": args.goal, "alt_states": alt}

    genes = load_genes(args.genes_file)
    state_embs = None
    if args.state_embs_file:
        import torch
        state_embs = torch.load(args.state_embs_file, weights_only=False)
        print(f"loaded state_embs: {args.state_embs_file} "
              f"({len(state_embs)} states)", flush=True)
    n_start = sum(1 for v in ds[args.label] if str(v) == args.start)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"model": args.model_dir, "model_type": args.model_type,
                      "start": args.start, "goal": args.goal, "alt_states": alt,
                      "n_start_cells": n_start, "n_genes": len(genes)}, indent=2), flush=True)

    results = []
    for sym, ens in genes:
        tag = f"{args.label}_start{args.start}_goal{args.goal}_{args.perturb_type}_{sym}"
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            isp = InSilicoPerturber(
                perturb_type=args.perturb_type,
                genes_to_perturb=[ens],          # 群摂動を避けるため 1 遺伝子ずつ
                combos=0,
                anchor_gene=None,
                model_type=args.model_type,
                num_classes=args.num_classes,
                emb_mode=args.emb_mode,
                cell_emb_style="mean_pool",
                filter_data=None,                # 開始状態の絞り込みは cell_states_to_model が行う
                cell_states_to_model=states,
                state_embs_dict=state_embs,
                max_ncells=(None if args.max_ncells in (0, -1) else args.max_ncells),
                cell_inds_to_perturb="all",
                emb_layer=args.emb_layer,
                forward_batch_size=args.forward_batch_size,
                nproc=args.nproc,
                token_dictionary_file=args.token_dict,
            )
            print(f"\n=== {args.perturb_type} {sym} ({ens}) start={args.start} ===", flush=True)
            isp.perturb_data(
                model_directory=args.model_dir,
                input_data_file=args.input_dataset,
                # 末尾スラッシュ必須: ISP は f"{output_directory}in_silico_..." で
                # パスを組み立てるため、無いと親ディレクトリに平坦に出力される
                output_directory=str(out_dir) + "/",
                output_prefix=tag,
            )
            results.append({"symbol": sym, "ensembl": ens, "ok": True, "dir": str(out_dir)})
        except Exception as exc:                 # 1 遺伝子の失敗で全体を止めない
            print(f"!! {sym} ({ens}) failed: {exc}", flush=True)
            traceback.print_exc()
            results.append({"symbol": sym, "ensembl": ens, "ok": False, "error": str(exc)})

    ok = [r for r in results if r["ok"]]
    summary = out_root / f"{args.label}_start{args.start}_goal{args.goal}_run_summary.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ndone: {len(ok)}/{len(results)} genes succeeded -> {summary}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
