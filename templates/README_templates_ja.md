# マウス Geneformer テンプレート集（notebook → Python スクリプト）

`Mouse-Geneformer-MPS` の notebook 群で示されている手順を、再利用できる CLI スクリプトに
書き直したもの。**リポジトリ本体（`geneformer/`, 重み, notebook）は変更していない。**

対象 notebook:
- `execute_tokenizer.py` → `template_tokenize.py`
- `cell_type_extract_and_plot_cell_embeddings_1.ipynb` → `template_embed.py`
- `cell_classification.ipynb` → `template_finetune_classifier.py`
- `in_silico_perturbation.ipynb` → `template_isp_delete.py` / `template_isp_stats.py`

## 先に知っておくべきこと（この環境の落とし穴）

1. **モデルは V1 系**（`mouse-Geneformer-L12-E20`: hidden 256 / 12 層 / vocab 56,084 /
   `max_position_embeddings` 2048）。トークン辞書は **`<pad>` と `<mask>` のみで
   `<cls>` / `<eos>` を持たない**。したがって埋め込みモードは常に **`cell`**
   （`cls` は無い。上位版 Geneformer V2 とはここが違う）。
2. **遺伝子 ID は ENSMUSG**。`genes_to_perturb` / トークン辞書のキーは
   マウス Ensembl ID。遺伝子記号は `MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl`
   で変換する（このファイルは ISP 統計の出力名にも使われる）。
3. **パスが壊れている**。本リポジトリは `~/workspace/` から
   `~/workspace/research/` へ移動したが、
   `geneformer/tokenizer.py` の `GENE_MEDIAN_FILE` / `TOKEN_DICTIONARY_FILE`、
   `geneformer/in_silico_perturber_stats.py` の `GENE_NAME_ID_DICTIONARY_FILE`
   は**旧パスのまま**。テンプレートはすべてリポジトリ位置から実パスを自動解決して
   明示的に渡すので、そのまま使える。
4. **トークン化は `Path(data_directory).glob("*.h5ad")` でディレクトリ内の全 h5ad を処理**し、
   1 ファイルにつき `<prefix>_<番号>.dataset` を出す。
   → **入力 h5ad は専用ディレクトリに 1 つだけ置くこと。**
5. **`genes_to_perturb` のリストは「群摂動」**。事前フィルタで
   『リスト内の全遺伝子を含む細胞』が必要になり、
   `"No cells in dataset contain all genes to perturb as a group."` で落ちる。
   → **個別 KO は 1 遺伝子 = 1 ラン**（`template_isp_delete.py` がそう実装）。
6. **上位版の `Classifier` クラスは無い**。fine-tuning は notebook と同じく
   `BertForSequenceClassification` + `Trainer` + `DataCollatorForCellClassification`。
   `template_finetune_classifier.py` がそれを実装している。
7. `cell_classification.ipynb` には import typo がある
   （`from transformers.training_args import TraianingArguments`）→ 正しくは `TrainingArguments`。
   テンプレートでは修正済み。`fp16=True` も MPS では不安定なので既定 off。
8. goal-state shift の**状態埋め込みは ISP 内部で計算される**
   （上位版のような `state_embs_dict` 引数は無い）。

## 使い方

```bash
PY=~/workspace/research/Mouse-Geneformer-MPS/.venv/bin/python
T=~/workspace/research/Mouse-Geneformer-MPS/templates

# 1) トークン化（h5ad を 1 つだけ含むディレクトリを渡す）
$PY $T/template_tokenize.py \
    --input-dir  /path/to/mouse_input/ \
    --output-dir /path/to/outputs/tokenized \
    --prefix GSE273983_mouse \
    --attrs leiden region_cluster_corr

# 2) 埋め込み（frozen）
$PY $T/template_embed.py \
    --model-dir ~/workspace/research/Mouse-Geneformer-MPS/mouse-Geneformer-L12-E20 \
    --input-dataset /path/to/outputs/tokenized/GSE273983_mouse_0.dataset \
    --output-dir /path/to/outputs/embeddings --prefix pretrained_cell_embeddings \
    --labels leiden --max-ncells 0

# 3) fine-tuning（cell classifier）
$PY $T/template_finetune_classifier.py \
    --model-dir ~/workspace/research/Mouse-Geneformer-MPS/mouse-Geneformer-L12-E20 \
    --input-dataset /path/to/outputs/tokenized/GSE273983_mouse_0.dataset \
    --output-dir /path/to/outputs/runs/cellClassifier_leiden \
    --label-col leiden --epochs 1 --batch-size 8

# 4) in silico deletion（遺伝子ごとに 1 ラン）
$PY $T/template_isp_delete.py \
    --model-dir /path/to/outputs/runs/cellClassifier_leiden \
    --model-type CellClassifier --num-classes 9 \
    --input-dataset /path/to/outputs/tokenized/GSE273983_mouse_0.dataset \
    --output-dir /path/to/outputs/isp --genes-file /path/to/genes_mouse.json \
    --label leiden --start 6 --goal 0

# 5) 統計（goal-state shift）
$PY $T/template_isp_stats.py --each-subdir \
    --input-dir /path/to/outputs/isp --glob 'leiden_start6_goal0_delete_*' \
    --output-dir /path/to/outputs/isp_stats --prefix leiden_start6_goal0 \
    --label leiden --start 6 --goal 0
```

## 各テンプレートの引数

- `template_tokenize.py` — `--input-dir`（h5ad 1 つだけ）/ `--output-dir` / `--prefix` /
  `--attrs`（持ち越す obs 列）/ `--nproc` / `--file-format`
- `template_embed.py` — `--model-dir` / `--input-dataset` / `--output-dir` / `--prefix` /
  `--model-type` / `--num-classes` / `--emb-mode cell` / `--emb-layer` / `--labels` /
  `--max-ncells`（**既定 1000 なので 0 で全細胞**）
- `template_finetune_classifier.py` — `--label-col` / `--test-size` / `--epochs` /
  `--batch-size` / `--lr` / `--freeze-layers` / `--pad-to-max` / `--fp16` / `--seed`
  → 出力先ディレクトリはそのまま `model_type="CellClassifier"` として読める
- `template_isp_delete.py` — `--genes-file`（`{symbol: {ensembl: ...}}` かリスト）/
  `--model-type` / `--num-classes` / `--label` / `--start` / `--goal` / `--alt` /
  `--perturb-type`（delete/overexpress/inhibit/activate）
- `template_isp_stats.py` — `--mode goal_state_shift` / `--null-dir`（渡すと p 値・FDR）/
  `--each-subdir` + `--glob` で遺伝子別 run を一括処理

## 実行時の互換・性能上の注意（検証済み）

### リポジトリ側に入れた追加修正

9. **`EncoderOnlyModel`（高速化）** — 埋め込み抽出・ISP は `outputs.hidden_states[layer]`
   しか使わず `.logits` は未使用なのに、`BertForMaskedLM` は毎 forward で語彙 56,084 への
   射影を計算して捨てていた。`load_model()` が encoder のみを呼ぶ薄いラッパーを返すように
   して、forward 時間を約 40% 削減（MPS: 35 ms → 21 ms / batch8@len619）。出力は同一。
10. **ISP に `state_embs_dict` 引数を追加（キャッシュ）** — 元実装は遺伝子ごとに
   全クラスの状態平均埋め込みを再計算していた（17,533 細胞 × 9 状態）。事前計算した
   dict を渡せるようにし、90 ランで再利用 → **147 秒/遺伝子 → 14.5 秒/遺伝子**。
11. **`safe_num_proc()`（MPS fork クラッシュ回避）** — MPS 初期化後に `datasets` の
   `map`/`filter` が子プロセスを fork すると
   `"One of the subprocesses has abruptly died during map operation."` で落ちる。
   `num_proc` が 1 以下のときは単一プロセス（None）に落とすよう修正。

### MPS での fine-tuning（重要）

12. **可変長バッチで MPS のアロケータキャッシュが増え続け、システムメモリを枯渇させる**。
    実測では step 200 付近で空きメモリが数十 MB まで崩壊し、step 時間が 0.33 s → 20-40 s に
    劣化した（スワップ落ち）。テンプレートは `--empty-cache-every`（既定 10 step）で
    `torch.mps.empty_cache()` を呼ぶ `TrainerCallback` を登録して回避する。
    **0 にすると現象が再発するので注意。**
13. **`Trainer.save_model()` は collator の precollator を tokenizer として保存しようとして
    `AttributeError` になる**（`PrecollatorForGeneAndCellClassification has no attribute
    save_pretrained`）。テンプレートは `save_strategy="no"` にして学習後に
    `model.save_pretrained()` を直接呼ぶ。
14. **DataLoader の worker は使わない**（`--nproc 0` 推奨）。8 worker では各ワーカーが
    データセットのコピー（数百 MB）を保持し、メモリ圧迫を悪化させる。
15. **`datasets 5.0.1` の `Dataset.train_test_split` は `stratify_by` 非対応**。
    テンプレートは sklearn で層化分割している。
16. **正常時は進捗バーが出ないことがある**。tqdm の `\r` 更新はログへリダイレクトすると
    バッファされ、「停止したように見える」。`python -u` を付けても完全ではないので、
    **経過時間と CPU 使用率で判断する**（実測: 埋め込み 17,533 細胞 ≈ 4 分、fine-tune
    2,796 step ≈ 16 分）。

## 検証状況

GSE273983 Paneth 細胞データ（マウス scRNA-seq）で全テンプレートを実行済み。詳細は
`~/workspace/opencodews/data/GSE273983/geneformer_mouse/README.md` を参照。
