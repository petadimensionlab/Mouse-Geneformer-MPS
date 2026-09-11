# Mouse-Geneformer MPS compatible update

Writer: Shinji Nakaoka

## Note

This repository contains the source code of mouse-Geneformer to work with Apple Silicon GPU (MPS). 

## Pretrained Models

- [mouse-Geneformer (base model)](https://drive.google.com/file/d/1gM3gcc3DlNGt5bAcqHbeRxtdMktGeDEg/view?usp=sharing)
- [mouse-Geneformer-12L-E20 (large model)](https://drive.google.com/file/d/1xKMyFA4JJeRigcJPsU2XNyxEW25Q247u/view?usp=sharing)

## Files You Must Download (Not Included in This Repository)

The following files are **NOT included** in this repository due to size and licensing. You must download them separately.

### 1. Pretrained Model Files

| File | Source | Save to |
|------|--------|---------|
| `mouse-Geneformer` (base model) | [Google Drive](https://drive.google.com/file/d/1gM3gcc3DlNGt5bAcqHbeRxtdMktGeDEg/view?usp=sharing) | `./mouse-Geneformer/` |
| `mouse-Geneformer-12L-E20` (large model) | [Google Drive](https://drive.google.com/file/d/1xKMyFA4JJeRigcJPsU2XNyxEW25Q247u/view?usp=sharing) | `./mouse-Geneformer-L12-E20/` |

### 2. Token Dictionary & Gene Median Dictionary

Download from HuggingFace: https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/tree/main

| File | Purpose | Default Config Path |
|------|---------|-------------------|
| `MLM-re_token_dictionary_v1.pkl` | Token dictionary (Ensembl ID→token) | `TOKEN_DICTIONARY_FILE` |
| `mouse_gene_median_dictionary.pkl` | Gene median expression values | `GENE_MEDIAN_FILE` |
| `MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl` | Gene symbol mapping (optional) | Used by `in_silico_perturber_stats` |

### 3. Evaluation Dataset (for in silico perturbation)

Choose one or more from `eval_dataset/` on HuggingFace:

| Dataset | Description |
|---------|-------------|
| `kidney_isp_mouse_tokenize_dataset_v-n1.dataset` | Kidney disease model (default in notebooks) |
| `Cop1KO_isp_mouse_tokenize_dataset_v-n1.dataset` | COP1 knockout microglia |
| `EX_to_human_from_mouse_covid19_blood_isp_mouse_tokenize_dataset_v-n1.dataset` | COVID-19 (cross-species) |
| `EX_to_human_from_mouse_myocardial_infarction_heart_isp_mouse_tokenize_dataset_v-n1.dataset` | Myocardial infarction (cross-species) |
| `all_organ_normal_mouse_tokenize_easy_dataset_v-n1.dataset` | Cell type classification (easy) |
| `all_organ_normal_mouse_tokenize_hard_dataset_v-n1.dataset` | Cell type classification (hard) |

### Automated Download Script

```python
from huggingface_hub import snapshot_download

# Download token dictionaries + kidney in silico perturbation dataset
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

The paths are now automatically resolved relative to the project directory — no manual path editing needed.

## Setup

### Requirements

- Python 3.12+
- PyTorch 2.2+ (with MPS or CUDA support)
- Apple Silicon (MPS) recommended for local execution

### Installation

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

The project now uses **numpy 2+** and **torch 2.2+** (upgraded from the original torch 2.0.1 for Python 3.12 compatibility). TensorFlow has been removed as it was not used by the project code.

### Device Configuration

The device is automatically detected (MPS → CUDA → CPU):

```python
from geneformer.tokenizer import DEVICE
print(DEVICE)  # "mps" on Apple Silicon, "cuda:0" on NVIDIA, "cpu" otherwise
```

### Automatic Resource Tuning

The `InSilicoPerturber` and `EmbExtractor` classes automatically tune batch size and CPU parallelism based on available hardware:

- **`forward_batch_size`**: Calculated from free VRAM (~350 MiB per sample at 2048 tokens, cap 250)
- **`nproc`**: Set to half of available CPU cores

Tensor Cores are automatically enabled on compatible NVIDIA GPUs (`torch.set_float32_matmul_precision("high")`).

To use these defaults, simply omit `forward_batch_size` and `nproc` when constructing the class. Explicit values override auto-tuning.

## Usage

### In Silico Perturbation

The notebook `in_silico_perturbation.ipynb` performs in silico perturbation experiments using a pretrained or fine-tuned model.

Key steps:
1. Set the `input_data_file` path to your downloaded dataset
2. Configure `start_state` and `end_state` based on your dataset's disease labels
3. Run the perturbation analysis

```python
isp.perturb_data(
    "/path/to/model/",
    dataset_name,  # your dataset
    "./results/",
    "output_prefix"
)
```

### Cell Classification

The notebook `cell_classification.ipynb` performs cell type classification by fine-tuning the pretrained model.

## Notes on Datasets and Paths

This project relies on several data files that must be downloaded separately:

| File | Config Variable | Source |
|------|---------------|--------|
| Token dictionary | `TOKEN_DICTIONARY_FILE` | HuggingFace dataset |
| Gene median dictionary | `GENE_MEDIAN_FILE` | HuggingFace dataset |
| Evaluation dataset | Set directly in notebook | HuggingFace dataset `eval_dataset/` |
| Pretrained model | Set in notebook | Google Drive |

All paths are now automatically resolved relative to the project directory.

## Changelog

### v0.3.0 — h5ad Tokenization Support

- **`execute_tokenizer.py`**: Now runs with **h5ad** input instead of loom
  - `file_format="h5ad"` with `data_directory="./data/tutorial_h5ad/"`, `output_directory="./data/tokenized/"`, `output_prefix="tutorial_mouse"`
  - `nproc` auto-set to `min(8, os.cpu_count())`
- **Tutorial data**: Uses `scanpy.datasets.paul15()` (mouse bone marrow, 2,730 cells × 3,451 genes)
  - Gene symbols mapped to mouse Ensembl IDs (`ENSMUSG...`) using the project's `GeneSymbol_to_EnsemblID.pkl` (3,107 genes matched the token dictionary)
  - `obs["n_counts"]` computed from raw counts
- **`geneformer/tokenizer.py` h5ad path fixes** (h5ad path was previously non-functional):
  - `self.genelist_dict` now initialized in `__init__` (was only set in the loom path → `AttributeError` on h5ad-only runs)
  - `ad.read()` (deprecated) → `ad.read_h5ad()`
  - Pandas label-vs-positional indexing: `adata.var["ensembl_id"][loc]` → `.iloc[loc]`
  - `tokenize_anndata()` return value count fixed (2 → 3) to match `tokenize_files()` unpacking
  - File count glob fixed from hardcoded `*.loom` to `*.{file_format}` (h5ad runs no longer infinite-loop / fail to terminate)

### v0.2.0 — Path Portability, Auto-Resource Tuning, and Build Fixes

**Portable Path Resolution**
- `geneformer/tokenizer.py`: Hardcoded absolute paths replaced with `Path(__file__).parent.parent` — files are now found relative to the project directory, no manual path editing needed after download
- `geneformer/in_silico_perturber_stats.py`: `GENE_NAME_ID_DICTIONARY_FILE` path similarly made portable
- `in_silico_perturbation.ipynb`: All hardcoded `/Users/petadimensionlab/...` paths replaced with relative paths (`./results/`, `./mouse-Geneformer-L12-E20/`)

**Automatic Resource Tuning** (NVIDIA GB10 / large-GPU optimized)
- `geneformer/in_silico_perturber.py`:
  - `auto_forward_batch_size()`: Calculates optimal batch size from available VRAM (~350 MiB per sample, capped at 250 for safe 2048-token sequences)
  - `auto_nproc()`: Uses half of available CPU cores (avoids system overload)
  - `torch.set_float32_matmul_precision("high")`: Enables Tensor Cores on compatible GPUs
  - `load_model()`: Simplified GPU placement
  - `InSilicoPerturber.__init__` defaults are overridden with auto-tuned values when defaults are used
- `geneformer/emb_extractor.py`: Same auto-tuning for `forward_batch_size` and `nproc`
- `in_silico_perturbation.ipynb`: Removed explicit `forward_batch_size` and `nproc` from `InSilicoPerturber` call (now uses auto-tuned defaults)

**Build System Fix**
- `pyproject.toml`: Added `[tool.setuptools.packages.find]` with `include = ["geneformer*"]` to fix setuptools flat-layout error caused by the `data/` directory

### Migration to NumPy 2+ and MPS Support

Key changes from the original codebase:

- **NumPy upgraded** from 1.24.3 to 2.x
- **PyTorch upgraded** from 2.0.1 to 2.2+ (required for Python 3.12 compatibility)
- **TensorFlow removed** (not imported by any project code)
- **Device auto-detection** added in `geneformer/tokenizer.py` — automatically selects MPS (Apple Silicon), CUDA, or CPU
- **`datasets` library compatibility**: Fixed `Column` object handling for datasets v5.x API changes
- **Notebooks**: Hardcoded `"cuda"` device strings replaced with dynamic `DEVICE` variable
- **`requirements.txt`** reduced from 227 pinned transitive dependencies to 27 top-level packages
- **Dependencies reorganized**: CUDA packages (`nvidia-*`) and DeepSpeed moved to optional extras

### Bug Fixes

- `geneformer/in_silico_perturber.py`:
  - `forward_pass_single_cell`: Fixed batch dimension (added `.unsqueeze(0)`)
  - `make_perturbation_batch`: Fixed Column multiplication error
  - `compute_batch_embeddings` / `get_cell_state_avg_embs`: Replaced `set_format(type="torch")` with explicit tensor conversion for datasets v5 compatibility
  - `cos_sim_shift`: Added automatic dimension unification — unsqueezes 2D `[seq, hidden]` tensors to 3D `[batch, seq, hidden]` before comparison, preventing shape mismatches when processing batched perturbations
  - `empty_cache()`: Replaced all `torch.cuda.empty_cache()` calls with an MPS-aware helper that handles both CUDA and MPS (`torch.mps.empty_cache()`)
  - Fixed typo: `"input_ids "` → `"input_ids"` (extra space in key name)
- `geneformer/pretrainer.py`: Graceful handling of missing token dictionary file (uses minimal placeholder)
- `geneformer/emb_extractor.py`: Same `set_format` → explicit tensor conversion fixes; same `torch.cuda.empty_cache()` → `empty_cache()` replacement
- `geneformer/__init__.py`: Removed imports of non-existent classes (`Cell_Type_Classification_TranscriptomeTokenizer`, `In_Silico_TranscriptomeTokenizer`)
- `geneformer/tokenizer.py`: Fixed escape sequence in regex (raw string `r"\(|,"`)
- `geneformer/tokenizer.py`: Updated hardcoded data paths from old workspace (`zedws/Mouse-Geneformer`) to current workspace (`Mouse-Geneformer-MPS`)
- `geneformer/in_silico_perturber_stats.py`: Fixed placeholder path `GENE_NAME_ID_DICTIONARY_FILE` (`/path/to/save/...`) to actual data directory
- `in_silico_perturbation.ipynb`: Fixed missing leading `/` in output path passed to `ispstats.get_stats()`
