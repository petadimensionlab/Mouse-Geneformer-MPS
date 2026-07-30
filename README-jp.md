# Mouse-Geneformer (日本語)

著者: Keita Ito

## 概要

本リポジトリは、マウス単一細胞 RNA シーケンスデータ解析のための mouse-Geneformer のソースコードを提供します。Mouse-Geneformer は大規模マウス単一細胞データセット mouse-Genecorpus-20M で事前学習されたモデルであり、マウス遺伝子ネットワークの解析に特化しています。細胞型分類の精度向上と、in silico 摂動実験による疾患原因遺伝子の同定を可能にします。

[[bioRxiv](https://www.biorxiv.org/content/10.1101/2024.09.09.611960v1)]

## 引用

```bibtex
@article{Ito2024.09.09.611960,
   author = {Ito, Keita and Hirakawa, Tsubasa and Shigenobu, Shuji and Fujiyoshi, Hironobu and Yamashita, Takayoshi},
   title = {Mouse-Geneformer: A Deep Leaning Model for Mouse Single-Cell Transcriptome and Its Cross-Species Utility},
   journal = {bioRxiv},
   year = {2024},
   URL = {https://www.biorxiv.org/content/early/2024/09/13/2024.09.09.611960}
}
```

## 事前学習済みモデル

- [mouse-Geneformer（ベースモデル）](https://drive.google.com/file/d/1gM3gcc3DlNGt5bAcqHbeRxtdMktGeDEg/view?usp=sharing)
- [mouse-Geneformer-12L-E20（大モデル）](https://drive.google.com/file/d/1xKMyFA4JJeRigcJPsU2XNyxEW25Q247u/view?usp=sharing)

## ダウンロードが必要なファイル（リポジトリに含まれていません）

以下のファイルはサイズ・ライセンスの都合上、このリポジトリには**含まれていません**。別途ダウンロードしてください。

### 1. 事前学習済みモデル

| ファイル | 入手先 | 保存先 |
|---------|--------|--------|
| `mouse-Geneformer`（ベースモデル） | [Google Drive](https://drive.google.com/file/d/1gM3gcc3DlNGt5bAcqHbeRxtdMktGeDEg/view?usp=sharing) | `./mouse-Geneformer/` |
| `mouse-Geneformer-12L-E20`（大モデル） | [Google Drive](https://drive.google.com/file/d/1xKMyFA4JJeRigcJPsU2XNyxEW25Q247u/view?usp=sharing) | `./mouse-Geneformer-L12-E20/` |

### 2. トークン辞書・遺伝子中央値辞書

HuggingFace からダウンロード: https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/tree/main

| ファイル | 目的 | 設定変数 |
|---------|------|---------|
| `MLM-re_token_dictionary_v1.pkl` | トークン辞書（Ensembl ID→token） | `TOKEN_DICTIONARY_FILE` |
| `mouse_gene_median_dictionary.pkl` | 遺伝子中央値発現量 | `GENE_MEDIAN_FILE` |
| `MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl` | 遺伝子名変換（オプション） | `in_silico_perturber_stats` で使用 |

### 3. 評価データセット（in silico perturbation 用）

HuggingFace の `eval_dataset/` から1つ以上を選択:

| データセット | 説明 |
|------------|------|
| `kidney_isp_mouse_tokenize_dataset_v-n1.dataset` | 腎臓疾患モデル（ノートブックのデフォルト） |
| `Cop1KO_isp_mouse_tokenize_dataset_v-n1.dataset` | COP1 ノックアウト（ミクログリア） |
| `EX_to_human_from_mouse_covid19_blood_isp_mouse_tokenize_dataset_v-n1.dataset` | COVID-19（ヒトへの応用） |
| `EX_to_human_from_mouse_myocardial_infarction_heart_isp_mouse_tokenize_dataset_v-n1.dataset` | 心筋梗塞（ヒトへの応用） |
| `all_organ_normal_mouse_tokenize_easy_dataset_v-n1.dataset` | 細胞型分類（易） |
| `all_organ_normal_mouse_tokenize_hard_dataset_v-n1.dataset` | 細胞型分類（難） |

### 自動ダウンロードスクリプト

```python
from huggingface_hub import snapshot_download

# トークン辞書 + 腎臓 in silico perturbation データセットをダウンロード
snapshot_download(
    repo_id="MPRG/Mouse-Genecorpus-20M",
    repo_type="dataset",
    allow_patterns=[
        "eval_dataset/in_silico_perturbation/kidney_isp*",
        "MLM-re_token_dictionary_v1.pkl",
        "mouse_gene_median_dictionary.pkl",
        "MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl",
    ],
    local_dir="./data/Mouse-Genecorpus-20M",
)
```

パスはプロジェクトディレクトリからの相対パスで自動解決されるため、手動でのパス編集は不要です。

## セットアップ

### 要件

- Python 3.12+
- PyTorch 2.2+（MPS または CUDA 対応）
- Apple Silicon（MPS）でのローカル実行を推奨

### インストール

```bash
# 仮想環境を作成（推奨）
python3 -m venv .venv
source .venv/bin/activate

# 依存関係をインストール
pip install -e ".[dev]"
```

本プロジェクトは **numpy 2+** および **torch 2.2+** を使用しています（Python 3.12 互換性のため元の torch 2.0.1 からアップグレード）。TensorFlow はプロジェクトコードで使用されていないため削除されました。

### デバイス設定

デバイスは自動検出されます（MPS → CUDA → CPU）:

```python
from geneformer.tokenizer import DEVICE
print(DEVICE)  # Apple Silicon では "mps"、NVIDIA GPU では "cuda:0"、それ以外では "cpu"
```

### 自動リソースチューニング

`InSilicoPerturber` および `EmbExtractor` クラスは、利用可能なハードウェアに基づいてバッチサイズと CPU 並列数を自動調整します:

- **`forward_batch_size`**: 空き VRAM から計算（2048トークンでサンプルあたり約350 MiB、上限250）
- **`nproc`**: 使用可能な CPU コアの半分を使用

Tensor Core は互換性のある NVIDIA GPU で自動有効化されます（`torch.set_float32_matmul_precision("high")`）。

デフォルト値を使用するには、`forward_batch_size` と `nproc` を省略してクラスを構築してください。明示的な値を指定すると自動チューニングは適用されません。

## 使用方法

### In Silico 摂動実験

`in_silico_perturbation.ipynb` ノートブックで、事前学習モデルまたはファインチューニング済みモデルを使用した in silico 摂動実験を実行できます。

主な手順:
1. `input_data_file` パスをダウンロードしたデータセットに設定
2. データセットの疾患ラベルに基づいて `start_state` と `end_state` を設定
3. 摂動解析を実行

```python
isp.perturb_data(
    "/path/to/model/",
    dataset_name,  # データセット
    "./results/",
    "output_prefix"
)
```

### 細胞型分類

`cell_classification.ipynb` ノートブックで、事前学習モデルをファインチューニングして細胞型分類を実行できます。

## データファイルとパスについて

このプロジェクトは、別途ダウンロードが必要な以下のデータファイルに依存しています:

| ファイル | 設定変数 | 入手先 |
|---------|---------|--------|
| トークン辞書 | `TOKEN_DICTIONARY_FILE` | HuggingFace データセット |
| 遺伝子中央値辞書 | `GENE_MEDIAN_FILE` | HuggingFace データセット |
| 評価データセット | ノートブック内で直接指定 | HuggingFace データセット `eval_dataset/` |
| 事前学習モデル | ノートブック内で指定 | Google Drive |

すべてのパスはプロジェクトディレクトリからの相対パスで自動解決されます。

## 変更履歴

### v0.2.0 — パスのポータビリティ対応、自動リソースチューニング、ビルド修正

**パスのポータビリティ対応**
- `geneformer/tokenizer.py`: ハードコードされた絶対パスを `Path(__file__).parent.parent` を用いた相対パスに変更 — ダウンロード後の手動パス編集は不要
- `geneformer/in_silico_perturber_stats.py`: `GENE_NAME_ID_DICTIONARY_FILE` のパスも同様にポータブル化
- `in_silico_perturbation.ipynb`: ハードコードされた `/Users/petadimensionlab/...` パスをすべて相対パス（`./results/`、`./mouse-Geneformer-L12-E20/`）に置き換え

**自動リソースチューニング**（NVIDIA GB10 / 大容量GPU向け最適化）
- `geneformer/in_silico_perturber.py`:
  - `auto_forward_batch_size()`: 空きVRAMから最適なバッチサイズを算出（サンプルあたり約350 MiB、上限250）
  - `auto_nproc()`: 使用可能なCPUコアの半分を使用（システム過負荷防止）
  - `torch.set_float32_matmul_precision("high")`: 互換性のあるNVIDIA GPUでTensor Coreを有効化
  - `load_model()`: GPU配置を簡略化
  - `InSilicoPerturber.__init__`: デフォルト値使用時に自動チューニング値を適用
- `geneformer/emb_extractor.py`: `forward_batch_size` と `nproc` に同様の自動チューニングを適用
- `in_silico_perturbation.ipynb`: `InSilicoPerturber` 呼び出しから明示的な `forward_batch_size` と `nproc` を削除（自動チューニングを使用）

**ビルドシステム修正**
- `pyproject.toml`: `[tool.setuptools.packages.find]` に `include = ["geneformer*"]` を追加し、`data/` ディレクトリに起因する setuptools の flat-layout エラーを修正

### NumPy 2+ への移行と MPS サポート

元のコードベースからの主な変更点:

- **NumPy**: 1.24.3 から 2.x へアップグレード
- **PyTorch**: 2.0.1 から 2.2+ へアップグレード（Python 3.12 互換性のため）
- **TensorFlow**: 削除（プロジェクトコードで使用されていないため）
- **デバイス自動検出**: `geneformer/tokenizer.py` に追加 — MPS（Apple Silicon）、CUDA、CPU を自動選択
- **datasets ライブラリ互換性**: datasets v5.x の API 変更に伴う Column オブジェクト対応を修正
- **ノートブック**: ハードコードされた `"cuda"` デバイス文字列を動的 `DEVICE` 変数に置き換え
- **requirements.txt**: 227 の固定された推移的依存関係から 27 のトップレベルパッケージに削減
- **依存関係の再編成**: CUDA パッケージ（`nvidia-*`）と DeepSpeed をオプションの依存関係に移動

### バグ修正

- `geneformer/in_silico_perturber.py`:
  - `forward_pass_single_cell`: バッチ次元の欠落を修正（`.unsqueeze(0)` を追加）
  - `make_perturbation_batch`: Column の乗算エラーを修正
  - `compute_batch_embeddings` / `get_cell_state_avg_embs`: datasets v5 互換性のため `set_format(type="torch")` を明示的な tensor 変換に置き換え
  - `cos_sim_shift`: 次元の自動統一を追加 — 2D `[seq, hidden]` を 3D `[batch, seq, hidden]` に unsqueeze してから比較することで、バッチ処理時のシェイプ不一致を防止
  - `empty_cache()`: 全ての `torch.cuda.empty_cache()` を MPS 対応のヘルパー関数に置き換え（CUDA と MPS 両対応）
  - タイポ修正: `"input_ids "` → `"input_ids"`（キー名の余分なスペース）
- `geneformer/pretrainer.py`: トークン辞書ファイルがない場合のグレースフルハンドリング（最小限のプレースホルダーを使用）
- `geneformer/emb_extractor.py`: 同上の `set_format` → 明示的な tensor 変換の修正、`torch.cuda.empty_cache()` → `empty_cache()` の置き換え
- `geneformer/__init__.py`: 存在しないクラスのインポートを削除（`Cell_Type_Classification_TranscriptomeTokenizer`、`In_Silico_TranscriptomeTokenizer`）
- `geneformer/in_silico_perturber.py`: 正規表現のエスケープシーケンスを修正（raw string `r"\(|,"`）
- `geneformer/tokenizer.py`: データパスを旧ワークスペース（`zedws/Mouse-Geneformer`）から現ワークスペース（`Mouse-Geneformer-MPS`）に更新
- `geneformer/in_silico_perturber_stats.py`: `GENE_NAME_ID_DICTIONARY_FILE` のプレースホルダパス（`/path/to/save/...`）を実際のデータディレクトリに修正
- `in_silico_perturbation.ipynb`: `ispstats.get_stats()` に渡す出力パスの先頭の `/` が欠落していた問題を修正
