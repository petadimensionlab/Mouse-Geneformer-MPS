#!/usr/bin/env python3
"""End-to-end check: does quantization preserve the CELL EMBEDDINGS Geneformer produces?

Runs the project's own get_embs() (emb_mode="cell", exactly the EmbExtractor path)
on the real tokenized dataset for fp32 and for each quantization scheme, then compares
the resulting per-cell embedding vectors.
"""
import sys, copy
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from datasets import load_from_disk

PROJECT = "/Users/petadimensionlab/workspace/research/Mouse-Geneformer-MPS"
sys.path.insert(0, PROJECT)
MODEL = f"{PROJECT}/mouse-Geneformer-L12-E20"
N_CELLS, FBS, LAYER = 256, 32, -1

import geneformer.emb_extractor as ee
import geneformer.in_silico_perturber as isp
ee.EMB_device = "cpu"
isp.ISP_device = "cpu"
from geneformer.emb_extractor import get_embs
from geneformer.in_silico_perturber import quant_layers
from transformers import BertForMaskedLM

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
                    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])

def q_int(w, bits):
    qmax = 2 ** (bits - 1) - 1
    s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(w / s).clamp(-qmax - 1, qmax) * s

def q_nf4(w, block=64):
    sh = w.shape
    flat = w.reshape(-1, block)
    ab = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    n = flat / ab
    idx = torch.zeros_like(n, dtype=torch.long)
    best = torch.full_like(n, float("inf"))
    for i, c in enumerate(NF4):
        d = (n - c).abs()
        m = d < best
        idx = torch.where(m, torch.full_like(idx, i), idx)
        best = torch.where(m, d, best)
    return (NF4[idx] * ab).reshape(sh)

base = BertForMaskedLM.from_pretrained(MODEL, output_hidden_states=True, output_attentions=False)
base.eval()
data = load_from_disk(f"{PROJECT}/data/tokenized/tutorial_mouse_0.dataset")
print(f"tokenized dataset: {len(data)} cells, features {list(data.features)}")
data = data.select(range(min(N_CELLS, len(data))))
L2Q = quant_layers(base) + LAYER
print(f"emb_layer -> hidden_states[{L2Q}] (quant_layers={quant_layers(base)}); "
      f"pad_token_id={base.config.pad_token_id}")

def variant_dtype(model, dt):
    return copy.deepcopy(model).to(dt)

def variant_fq(model, emb_bits, lin_scheme, untie=False, cast=None):
    m = copy.deepcopy(model)
    if cast:
        m = m.to(cast)
    if emb_bits:
        w = m.get_input_embeddings().weight.data
        m.get_input_embeddings().weight.data = q_int(w, emb_bits) if emb_bits == 8 else q_nf4(w)
    if untie:
        m.get_output_embeddings().weight = nn.Parameter(
            m.get_input_embeddings().weight.data.clone())
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            if name.endswith("decoder") and not untie:
                continue
            if lin_scheme == 8:
                mod.weight.data = q_int(mod.weight.data, 8)
            elif lin_scheme == 4:
                mod.weight.data = q_nf4(mod.weight.data)
    return m

def cell_embs(m):
    if m.dtype != torch.float32:
        m = m.float()
    return get_embs(m, data, "cell", L2Q, base.config.pad_token_id, FBS, None)

ref = cell_embs(base).numpy()
print(f"reference cell embeddings: {ref.shape}")

def cos_rows(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

def simmat(x):
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return xn @ xn.T

ref_sim = simmat(ref)
iu = np.triu_indices(len(ref), 1)

VARIANTS = [
    ("fp16 whole model",            lambda: variant_dtype(base, torch.float16)),
    ("bf16 whole model",            lambda: variant_dtype(base, torch.bfloat16)),
    ("int8 encoder | fp32 emb",     lambda: variant_fq(base, None, 8, untie=True)),
    ("nf4 encoder  | fp32 emb",     lambda: variant_fq(base, None, 4, untie=True)),
    ("int8 encoder | int8 emb",     lambda: variant_fq(base, 8, 8)),
    ("int8 encoder | nf4 emb",      lambda: variant_fq(base, 4, 8)),
    ("nf4 encoder  | nf4 emb",      lambda: variant_fq(base, 4, 4)),
]

print(f"\n  {'variant':28s} {'cos mean':>9s} {'cos min':>8s} {'cos p5':>8s} "
      f"{'cell-sim rho':>13s} {'norm ratio':>11s}")
rows = []
for tag, build in VARIANTS:
    m = build()
    e = cell_embs(m).numpy()
    c = cos_rows(ref, e)
    rho = spearmanr(ref_sim[iu], simmat(e)[iu]).statistic
    nr = float(np.linalg.norm(e, axis=1).mean() / np.linalg.norm(ref, axis=1).mean())
    rows.append((tag, c.mean(), c.min(), np.percentile(c, 5), rho, nr))
    print(f"  {tag:28s} {c.mean():9.5f} {c.min():8.4f} {np.percentile(c,5):8.4f} "
          f"{rho:13.5f} {nr:11.4f}")
    del m

# how much does the *ranking* of cells in a perturbation-style score survive?
print("\n  (cell-sim rho = Spearman correlation of all pairwise cell-cell similarities "
      "vs fp32; 1.0 = identical neighbourhood structure)")
np.save(f"{PROJECT}/quant/ref_cell_embs_fp32.npy", ref)
print("saved reference embeddings to quant/ref_cell_embs_fp32.npy")
print("done")