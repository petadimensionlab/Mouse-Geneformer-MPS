# Mouse-Geneformer

Writer: Keita Ito

## Abstract

This repository contains the source code of mouse-Geneformer for analyzing single-cell RNA-sequence data of mouse. Mouse-Geneformer is a model pre-trained on the large mouse single-cell dataset mouse-Genecorpus-20M and designed to map the mouse gene network. It improves the accuracy of cell type classification of mouse cells and enables in silico perturbation experiments on mouse specimens.

[[bioRxiv](https://www.biorxiv.org/content/10.1101/2024.09.09.611960v1)]

## Citation

```bibtex
@article{Ito2024.09.09.611960,
   author = {Ito, Keita and Hirakawa, Tsubasa and Shigenobu, Shuji and Fujiyoshi, Hironobu and Yamashita, Takayoshi},
   title = {Mouse-Geneformer: A Deep Leaning Model for Mouse Single-Cell Transcriptome and Its Cross-Species Utility},
   journal = {bioRxiv},
   year = {2024},
   URL = {https://www.biorxiv.org/content/early/2024/09/13/2024.09.09.611960}
}
```

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

After downloading, update the paths in `geneformer/tokenizer.py` (lines 56-57) to match your local paths.

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

Update the paths in `geneformer/tokenizer.py` (lines 56-57) and the notebooks to point to your downloaded files.

## Changelog

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
  - `make_perturbation_batch`: Fixed Column multiplication erro
  - `compute_batch_embeddings` / `get_cell_state_avg_embs`: Replaced `set_format(type="torch")` with explicit tensor conversion for datasets v5 compatibility
  - Fixed typo: `"input_ids "` → `"input_ids"` (extra space in key name)
- `geneformer/pretrainer.py`: Graceful handling of missing token dictionary file (uses minimal placeholder)
- `geneformer/emb_extractor.py`: Same `set_format` → explicit tensor conversion fixes
- `geneformer/__init__.py`: Removed imports of non-existent classes (`Cell_Type_Classification_TranscriptomeTokenizer`, `In_Silico_TranscriptomeTokenizer`)
- `geneformer/tokenizer.py`: Fixed escape sequence in regex (raw string `r"\(|,"`)
- `geneformer/tokenizer.py`: Updated hardcoded data paths from old workspace (`zedws/Mouse-Geneformer`) to current workspace (`Mouse-Geneformer-MPS`)
- `geneformer/in_silico_perturber_stats.py`: Fixed placeholder path `GENE_NAME_ID_DICTIONARY_FILE` (`/path/to/save/...`) to actual data directory
- `in_silico_perturbation.ipynb`: Fixed missing leading `/` in output path passed to `ispstats.get_stats()`
