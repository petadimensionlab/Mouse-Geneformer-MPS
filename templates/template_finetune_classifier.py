#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マウス Geneformer セル分類 fine-tuning テンプレート（Classifier）
=================================================================

`cell_classification.ipynb` の流れを、再利用できる CLI スクリプトにしたもの。
マウス版 Geneformer には**上位版の `Classifier` クラスが無い**ため、
notebook と同じく `BertForSequenceClassification` + `Trainer` +
`DataCollatorForCellClassification` で自前学習する。

元 notebook からの主な修正:
  * `from transformers.training_args import TraianingArguments`（typo）を
    `TrainingArguments` に修正。
  * `fp16=True` は MPS で不安定なため既定 off（`--fp16` で有効化）。
  * チェックポイントパス等のベタ書きを CLI 引数化。
  * 全系列を max_position_embeddings までゼロ埋めする代わりに、
    「切り詰め + バッチ毎の動的パディング + group_by_length」を既定にした
    （MPS のメモリ効率が大幅に良い）。`--pad-to-max` で元 notebook と同じ挙動。

出力: <output-dir>/ に model(+config) / label_map.json / metrics.json / predictions.csv
      ISP の `model_type="CellClassifier"` からそのまま読める。

使用例:
  python template_finetune_classifier.py \
      --model-dir   /path/to/mouse-Geneformer-L12-E20 \
      --input-dataset /path/to/tokenized/GSE273983_mouse_0.dataset \
      --output-dir  /path/to/outputs/runs/cellClassifier_leiden \
      --label-col leiden --epochs 1 --batch-size 8 --test-size 0.15
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


def _make_mps_memory_callback(every_n_steps):
    """MPS のアロケータキャッシュ肥大を防ぐ TrainerCallback を作って返す。

    可変長バッチ（group_by_length + 動的パディング）で学習すると、バッチごとに
    テンソル形状が変わるため MPS のキャッシュが解放されず増え続ける。これがシステム
    メモリを食い潰し、スワップに落ちて step 時間が 0.3 s → 20-40 s に劣化する
    （実測: 空きメモリが数十 MB まで崩壊）。一定 step ごとに empty_cache で回避する。
    """
    from transformers import TrainerCallback

    class MPSMemoryCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            import torch
            if (state.global_step % every_n_steps == 0
                    and torch.backends.mps.is_available()):
                torch.mps.empty_cache()
            return control

    return MPSMemoryCallback()


def _compute_metrics(pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune mouse-Geneformer cell classifier")
    ap.add_argument("--model-dir", required=True, help="事前学習重み（mouse-Geneformer-L12-E20 等）")
    ap.add_argument("--input-dataset", required=True, help="tokenize 出力 *.dataset")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-col", required=True,
                    help="分類ラベルとして使う dataset 列（例: leiden）")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--lr-scheduler", default="linear")
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="先頭から何層の encoder を凍結するか（0 = 凍結なし）")
    ap.add_argument("--max-len", type=int, default=0,
                    help="切り詰める最大トークン長（0 = モデル config の max_position_embeddings）")
    ap.add_argument("--pad-to-max", action="store_true",
                    help="元 notebook と同じく全系列を max_len までゼロ埋めする")
    ap.add_argument("--fp16", action="store_true", help="MPS では非推奨")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--empty-cache-every", dest="empty_cache_every", type=int, default=10,
                    help="MPS のアロケータキャッシュを解放する間隔（step）。0 で無効")
    ap.add_argument("--attribute", default=None,
                    help="run 名に付ける接尾辞（例: ISP-Paneth）")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_from_disk
    from sklearn.metrics import confusion_matrix
    from transformers import BertForSequenceClassification, Trainer, TrainingArguments, set_seed
    from geneformer.collator_for_classification import DataCollatorForCellClassification

    set_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    ds = load_from_disk(args.input_dataset)
    if args.label_col not in ds.features:
        sys.exit(f"label column '{args.label_col}' not in {list(ds.features)}")

    # --- ラベルを整数化（leiden 等の文字列クラスを 0..K-1 に写像）---------------
    classes = sorted({str(v) for v in ds[args.label_col]})
    label_map = {c: i for i, c in enumerate(classes)}
    inv = {i: c for c, i in label_map.items()}
    ds = ds.map(lambda ex: {"label": label_map[str(ex[args.label_col])]},
                num_proc=max(1, args.nproc))
    print(f"classes ({len(classes)}): {label_map}", flush=True)

    # --- 学習/評価分割（個体情報が無いデータなので層化ランダム分割）--------------
    # datasets の train_test_split は版によって stratify_by 非対応のため sklearn を使う
    from sklearn.model_selection import train_test_split
    labels = np.asarray([int(v) for v in ds["label"]])
    idx_tr, idx_te = train_test_split(
        np.arange(len(ds)), test_size=args.test_size,
        random_state=args.seed, stratify=labels)
    train_ds, eval_ds = ds.select(idx_tr.tolist()), ds.select(idx_te.tolist())
    print(f"train/eval: {len(train_ds)} / {len(eval_ds)}", flush=True)

    # --- 切り詰め（任意でゼロ埋め）-----------------------------------------------
    cols = [c for c in ("input_ids", "label", "length") if c in train_ds.features]

    def _truncate(example):
        example["input_ids"] = example["input_ids"][:max_len]
        example["length"] = min(example.get("length", len(example["input_ids"])), max_len)
        return example

    def _pad(example):
        ids = example["input_ids"]
        example["input_ids"] = ids + [pad_id] * (max_len - len(ids))
        example["length"] = min(example["length"], max_len)
        return example

    model = BertForSequenceClassification.from_pretrained(
        args.model_dir, num_labels=len(classes),
        output_attentions=False, output_hidden_states=False,
        ignore_mismatched_sizes=True,
    ).to(device)
    max_len = args.max_len or model.config.max_position_embeddings
    pad_id = model.config.pad_token_id or 0
    print(f"max_len: {max_len} | pad_id: {pad_id}", flush=True)

    train_ds = train_ds.map(_truncate, num_proc=args.nproc)
    eval_ds = eval_ds.map(_truncate, num_proc=args.nproc)
    if args.pad_to_max:
        train_ds = train_ds.map(_pad, num_proc=args.nproc)
        eval_ds = eval_ds.map(_pad, num_proc=args.nproc)
    train_ds.set_format("torch", columns=cols)
    eval_ds.set_format("torch", columns=cols)

    # --- 層の凍結（任意）---------------------------------------------------------
    if args.freeze_layers > 0:
        for name, param in model.bert.named_parameters():
            if name.split(".")[0] == "encoder" and name.split(".")[2].isdigit():
                if int(name.split(".")[2]) < args.freeze_layers:
                    param.requires_grad = False
        print(f"froze first {args.freeze_layers} encoder layers", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_steps = max(1, len(train_ds) // max(1, args.batch_size) // 10)
    targs = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        seed=args.seed,
        do_train=True, do_eval=True,
        eval_strategy="epoch", save_strategy="no",
        logging_steps=log_steps,
        fp16=args.fp16,
        dataloader_num_workers=max(0, args.nproc),
        dataloader_pin_memory=args.nproc > 0,
        group_by_length=True, length_column_name="length",
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=False,
        eval_accumulation_steps=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=targs,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=_compute_metrics,
        callbacks=([_make_mps_memory_callback(args.empty_cache_every)]
                   if args.empty_cache_every > 0 else None),
    )
    trainer.train()
    preds = trainer.predict(eval_ds)

    # Trainer の save_model() は collator が持つ precollator を tokenizer として
    # 保存しようとして AttributeError になるため、モデルを直接保存する
    # （ISP は config.json + 重みがあれば読める）
    model.save_pretrained(str(out_dir))
    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(preds.metrics, indent=2), encoding="utf-8")

    y_true = np.asarray(preds.label_ids)
    y_pred = np.argmax(preds.predictions, axis=-1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    pd.DataFrame(cm, index=[f"true_{inv[i]}" for i in range(len(classes))],
                 columns=[f"pred_{inv[i]}" for i in range(len(classes))]
                 ).to_csv(out_dir / "confusion_matrix.csv")
    pd.DataFrame({"True": [inv[int(i)] for i in y_true],
                  "Pred": [inv[int(i)] for i in y_pred]}).to_csv(
        out_dir / "predictions.csv", index=False)

    print(json.dumps({k: v for k, v in preds.metrics.items()
                      if k in ("test_accuracy", "test_macro_f1")}, indent=2), flush=True)
    print(f"saved -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
