# Geneformer の重み量子化 — 調査と実測レポート

対象: `mouse-Geneformer-L12-E20` (このプロジェクトの学習済みモデル)
環境: macOS 26.6.2 arm64 / Python 3.14 / torch 2.13.0 / transformers 4.57.6 / MPS 利用可
(bitsandbytes・torchao・optimum は未インストール)

---

## 結論（先に）

1. **fp16/bf16 は事実上ロスレスで 2.00x 小さくなる。** 実チェックポイントを書き出して検証済み
   （細胞埋め込みのコサイン類似度 min 0.999999）。まずこれをやるべき。
2. **bitsandbytes の 4-bit（QLoRA 方式）はこのモデルには効きが悪い。** サイズは 1.92x しか
   縮まず、しかも埋め込み空間の劣化が最大（cos 0.944）。
3. **効くのは埋め込みテーブル側を量子化する方向。** エンコーダ int8 + 埋め込み int8 で
   **3.82x・cos 0.99945**（ほぼロスレス）。埋め込みのみ 4-bit にすると **5.39x・cos 0.99798**。
4. **エンコーダを 4-bit にするのが一番ダメージが大きい**（cos 0.944）。埋め込みテーブルは
   行間の冗長性が極めて高い（ランダム 400 遺伝子の平均ペア cos 0.4921）ので 4-bit に強い。
   → 一般的な QLoRA の直感（全体を 4-bit）と**逆**の配分が最適。

---

## 1. このモデルの構造 — なぜ一般的な量子化が効かないか

`config.json`: `BertForMaskedLM`, 12 layers, hidden 256, 4 heads, vocab 56,084, max_pos 2048, silu。
**`tie_word_embeddings: True`** — `cls.predictions.decoder.weight` は
`bert.embeddings.word_embeddings.weight` と同一テンソル。

| 構成要素 | パラメータ | 全体比 | fp32 サイズ |
|---|---:|---:|---:|
| word_embeddings（decoder と共有） | 14.358M | **67.3%** | 54.8 MiB |
| encoder Linear（attn/dense） | 6.291M | 29.5% | 24.0 MiB |
| position/token_type emb | 0.525M | 2.5% | 2.0 MiB |
| cls.* (MLM transform + bias) | 0.122M | 0.6% | 0.5 MiB |
| encoder bias / LayerNorm | 0.034M | 0.2% | 0.1 MiB |
| **合計** | **21.33M** | 100% | **81.4 MiB** |

**要点**: bitsandbytes や `torch.ao.quantization.quantize_dynamic` が触れるのは
`nn.Linear` だけ＝**全体の 29.5%**。残り 70.5%（埋め込み＋正規化）が fp32/fp16 のまま残る。
これが「4-bit にしたのに 4 倍しか縮まない」の正体。

### 上流の一次資料との整合

Theodoris 研の量子化論文（*Nature Computational Science* 2026,
doi:10.1038/s43588-026-00972-4）では、Geneformer を **8-bit で推論**、
**4-bit + LoRA でファインチューニング**し、フル精度と同等の zero-shot / few-shot /
fine-tuning 精度を保ちつつ **時間 15%・メモリ 34%**（同一バッチ、A100）を達成、
埋め込み空間の分離性と GATA4 の in silico 欠失も再現できたと報告している。

ただしこれは **GF-104M / GF-316M** での結果。GF-316M では語彙テーブル
(20,271 × 1152 = 23.4M) は**全パラメータの 7.4%** に過ぎず、量子化の効きが良い。
本モデルは**67.3%** が語彙テーブルなので、同じ手法では同じ効果は出ない。

---

## 2. 実測 — サイズ見積り（正確なビット予算）

| 方式 | サイズ | %fp32 | 倍率 |
|---|---:|---:|---:|
| fp32 baseline | 81.4 MiB | 100% | 1.00x |
| **fp16 / bf16（全体）** | **40.7 MiB** | 50.0% | **2.00x** |
| A: bnb int8（emb fp16 + decoder/encoder int8） | 51.7 MiB | 63.5% | 1.57x |
| A: bnb nf4（emb fp16 + decoder/encoder nf4） | 42.5 MiB | 52.2% | 1.92x |
| B: int8 emb + int8 encoder | 21.3 MiB | 26.2% | **3.82x** |
| B: int8 emb + nf4 encoder | 19.0 MiB | 23.3% | 4.28x |
| B: nf4 emb + int8 encoder | 15.1 MiB | 18.6% | **5.39x** |
| B: nf4 emb + nf4 encoder | 12.8 MiB | 15.7% | 6.36x |

（A は bitsandbytes の既定挙動の模擬。bnb は decoder を独立した Linear に切り離すため、
14.36M パラメータがもう 1 回ぶん加算される。**tied 重複の二重計上に注意**。）

---

## 3. 実測 — 忠実度その1: 合成バッチ（16 × 512 token）

| 方式 | hidden cos | cell-emb cos | logit maxdiff |
|---|---:|---:|---:|
| fp32 | 1.000000 | 1.000000 | 0 |
| A: bnb int8 | 0.999631 | 0.999655 | 2.51 |
| A: bnb nf4 | 0.944143 | 0.964620 | 22.6 |
| B: int8 emb + int8 encoder | 0.999503 | 0.999644 | 2.69 |
| B: int8 emb + nf4 encoder | 0.943967 | 0.964525 | 19.5 |
| B: nf4 emb + int8 encoder | 0.989390 | 0.998115 | 11.9 |
| B: nf4 emb + nf4 encoder | 0.933541 | 0.961163 | 23.5 |

---

## 4. 実測 — 忠実度その2: エンドツーエンド（本命）

このプロジェクトの `get_embs()`（`EmbExtractor` と同じ経路、`emb_mode="cell"`,
`hidden_states[11]`）を、実際のトークン化データ `data/tokenized/tutorial_mouse_0.dataset`
の 256 細胞で実行し、fp32 の細胞埋め込みと比較。

| 方式 | cos mean | cos min | cos p5 | cell-sim ρ | norm 比 |
|---|---:|---:|---:|---:|---:|
| fp16（全体） | **1.00000** | 1.0000 | 1.0000 | **1.00000** | 1.0000 |
| bf16（全体） | 0.99997 | 0.9999 | 1.0000 | 0.99993 | 1.0030 |
| int8 encoder / fp32 emb | 0.99946 | 0.9992 | 0.9993 | 0.99960 | 0.9940 |
| nf4 encoder / fp32 emb | 0.94424 | 0.9129 | 0.9237 | 0.96679 | 0.9579 |
| **int8 encoder / int8 emb** | **0.99945** | 0.9988 | 0.9993 | **0.99935** | 0.9933 |
| **int8 encoder / nf4 emb** | **0.99798** | 0.9877 | 0.9949 | 0.98024 | 0.9825 |
| nf4 encoder / nf4 emb | 0.94211 | 0.9108 | 0.9218 | 0.94825 | 0.9413 |

- `cell-sim ρ` = 全細胞ペアの類似度行列の Spearman 相関（近傍構造が保たれているか）。1.0 が完全一致。
- **エンコーダ 4-bit が支配的な誤差源**。埋め込みテーブルは 4-bit でもほぼ無傷。

---

## 5. 実測 — 速度と、使えない手法

バッチ 16 × 512 token の fwd 時間:

| デバイス / dtype | s/fwd | CPU 比 |
|---|---:|---:|
| CPU fp32 | 0.325 | 1.0x |
| MPS fp32 | 0.129 | **2.5x** |
| MPS fp16 | 0.137 | 2.4x |
| MPS bf16 | 0.137 | 2.4x |

- **`torch.ao.quantization.quantize_dynamic`（追加依存なしの int8）はこの Mac では動かない**:
  `RuntimeError: Didn't find engine for operation quantized::linear_prepack NoQEngine`
  — macOS arm64 の PyTorch には量子化 CPU カーネルが入っていない。しかも torch 2.13 で
  deprecated（torchao への移行推奨）。**この経路は諦めてよい。**
- エンコーダ重みの低ランク性は高くない（layer0 query の rank-8 でエネルギー 24.2%,
  layer11 で 32.2%）。LoRA を「重みの近似」として期待するのは筋が悪い（QLoRA の LoRA は
  更新側なので別の話）。

---

## 6. 推奨ルート（実装順）

### ルート 1: fp16 — 完了・検証済み ✅
`mouse-Geneformer-L12-E20-fp16/`（40.7 MiB, 2.00x, `model.safetensors`）と
`mouse-Geneformer-L12-E20-bf16/` を書き出し済み。標準の HF チェックポイントなので
Geneformer 側のコード変更は不要、読み込み時だけ dtype を指定する:

```python
model = BertForMaskedLM.from_pretrained(path, output_hidden_states=True,
                                        torch_dtype=torch.float16)   # or bfloat16
```
検証: 256 細胞の細胞埋め込み cos **min 0.999999**、最大要素差 4.5e-04。

### ルート 2: bitsandbytes 8-bit（論文と同じ系譜・要インストール）
bnb は PyPI で **macOS 14+ arm64 をサポート**（CPU ✅ / Metal MPS ✅、torch ≥ 2.9、
Python ≥ 3.10 — 本環境はすべて満たす）。0.50.0 で Apple Silicon バックエンドが大幅改善。

```bash
pip install -U bitsandbytes          # >= 0.50
```

```python
from transformers import BertForMaskedLM, BitsAndBytesConfig
import torch
cfg = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["cls.predictions.decoder"],   # tied テーブルは fp16 のまま残す
)
model = BertForMaskedLM.from_pretrained(MODEL_DIR, quantization_config=cfg,
                                        output_hidden_states=True)
model = model.to("mps")              # device_map ではなく明示的に移す方が安全
```
注意: 8-bit でも**サイズは 1.57x しか縮まない**（埋め込みが 67% を占めるため）。
利点は主にメモリではなく演算側。`Geneformer` 側は `load_model()` に
`quantization_config` を渡す改修が必要（`geneformer/in_silico_perturber.py:85`）。

### ルート 3: 埋め込み対応 int8 — 実圧縮の本命（自前実装）
3.82x・cos 0.99945。ライブラリでは得られないので自前で書く。方針:

1. `word_embeddings.weight` を **行ごと** の対称 int8（absmax/127）で保存。tied なので
   decoder は別途保存しない（1 テーブルだけ）。
2. encoder Linear の重みを **出力チャネルごと** int8 で保存。
3. bias / LayerNorm / position embedding は fp16 のまま。
4. 読み込み時に bf16 へ dequantize して通常の `BertForMaskedLM` として動かす
   （Geneformer 側の改修ゼロ）。

### ルート 4: 埋め込みだけ 4-bit — 最大圧縮
5.39x・cos 0.99798・cell-sim ρ 0.98024。スケールあたりの劣化が許容できるなら最良のトレードオフ。
NF4 コードブック + 64 要素ブロックの absmax で、埋め込みテーブルにのみ適用する。

### やらない方がよい
- **エンコーダへの 4-bit/NF4**（cos 0.944、ρ 0.948）。このモデルでは最大の誤差源。
- **`torch.ao` の dynamic int8**（この Mac では動かない・deprecated）。
- 埋め込みを触らずに bnb 4-bit（1.92x かつ cos 0.944 — 悪いとこ取り）。

---

## 7. 実務上の落とし穴

- **tied 重複**: `BertForMaskedLM` の `cls.predictions.decoder.weight` は
  `word_embeddings.weight` と共有。`named_parameters()` は既定で重複除去するが、
  `modules()` を回して `nn.Linear` を数えると 14.36M を二重計上する。
  bitsandbytes は読み込み時に tie を切って decoder を独立 Linear にするため、
  実サイズは素朴な見積りより増える。
- **`quant_layers()`**（`geneformer/in_silico_perturber.py:105`）は最大 layer index + 1 を返す
  ため `emb_layer=-1` で `hidden_states[11]` を見る。量子化後の層番号は変わらないが、
  bnb のように層を差し替える手法では `named_parameters()` の名前が変わる点に注意。
- **in silico perturbation は勾配を使う**（`ISP_device`）。bnb の `Linear4bit` も backward は
  通るが、MPS + 4-bit の逆伝播は検証が薄い。ルート 1/3 のような
  「dequantize して通常の fp16/bf16 モデルとして動かす」方式なら勾配は完全に無害。
- MPS は CPU の 2.5x 速い。量子化で精度を落とす前に、まず MPS + fp16 に寄せる価値がある。

---

## 8. 再現方法

```bash
cd ~/workspace/research/Mouse-Geneformer-MPS/quant
PY=~/workspace/research/Mouse-Geneformer-MPS/.venv/bin/python
$PY quant_report.py      # パラメータ内訳・忠実度・速度・失敗した経路
$PY quant_simulate.py    # 方式別のビット予算と忠実度
$PY e2e_embeddings.py    # 実データ 256 細胞での細胞埋め込み比較
$PY export_fp16.py       # fp16/bf16 チェックポイント書き出しと検証
```
出力は各スクリプトの `.out` に保存される。`quant/ref_cell_embs_fp32.npy` は
fp32 の基準細胞埋め込み（256 × 256）。

生成物: `mouse-Geneformer-L12-E20-fp16/`, `mouse-Geneformer-L12-E20-bf16/`（各 40.7 MiB）
