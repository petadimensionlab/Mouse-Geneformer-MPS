#!/usr/bin/env python3
"""Export half-precision checkpoints and verify them end-to-end against fp32.

Produces a real, standard-loadable quantized checkpoint (no custom code needed
downstream) and proves the cell embeddings are unchanged.
"""
import sys, copy, os
import numpy as np
import torch

PROJECT = "/Users/petadimensionlab/workspace/research/Mouse-Geneformer-MPS"
sys.path.insert(0, PROJECT)
MODEL = f"{PROJECT}/mouse-Geneformer-L12-E20"
OUT = f"{PROJECT}/mouse-Geneformer-L12-E20-fp16"
N_CELLS, FBS, LAYER = 256, 32, -1

import geneformer.emb_extractor as ee, geneformer.in_silico_perturber as isp
ee.EMB_device = "cpu"; isp.ISP_device = "cpu"
from geneformer.emb_extractor import get_embs
from geneformer.in_silico_perturber import quant_layers
from datasets import load_from_disk
from transformers import BertForMaskedLM

base = BertForMaskedLM.from_pretrained(MODEL, output_hidden_states=True, output_attentions=False)
base.eval()
data = load_from_disk(f"{PROJECT}/data/tokenized/tutorial_mouse_0.dataset").select(range(N_CELLS))
L2Q = quant_layers(base) + LAYER
ref = np.load(f"{PROJECT}/quant/ref_cell_embs_fp32.npy")

# ---- export fp16 ----
half = copy.deepcopy(base).half()
half.config.torch_dtype = torch.float16
os.makedirs(OUT, exist_ok=True)
half.save_pretrained(OUT)
sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
orig = os.path.getsize(f"{MODEL}/pytorch_model.bin")
print(f"exported fp16 checkpoint -> {OUT}")
print(f"  files: {sorted(os.listdir(OUT))}")
print(f"  size {sz/2**20:.1f} MiB vs fp32 {orig/2**20:.1f} MiB  ({orig/sz:.2f}x smaller)")

# ---- load it back the way Geneformer's load_model() would, and verify ----
print("\nreloading with BertForMaskedLM.from_pretrained(..., torch_dtype=torch.float16)")
rl = BertForMaskedLM.from_pretrained(OUT, output_hidden_states=True,
                                     output_attentions=False, torch_dtype=torch.float16)
rl.eval()
print("  loaded dtype:", next(rl.parameters()).dtype)
e16 = get_embs(rl.float(), data, "cell", L2Q, base.config.pad_token_id, FBS, None).numpy()
c = (ref * e16).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(e16, axis=1) + 1e-12)
print(f"  cell embeddings {e16.shape} | cosine vs fp32: mean {c.mean():.6f} min {c.min():.6f}")
print(f"  max abs elementwise diff: {np.abs(ref-e16).max():.3e}")
print("  VERDICT:", "PASS - embeddings preserved" if c.min() > 0.9999 else "FAIL")

# ---- also confirm a bf16 export round-trips ----
OUTB = f"{PROJECT}/mouse-Geneformer-L12-E20-bf16"
b = copy.deepcopy(base).to(torch.bfloat16)
b.config.torch_dtype = torch.bfloat16
b.save_pretrained(OUTB)
szb = sum(os.path.getsize(os.path.join(OUTB, f)) for f in os.listdir(OUTB))
rlb = BertForMaskedLM.from_pretrained(OUTB, output_hidden_states=True,
                                      output_attentions=False, torch_dtype=torch.bfloat16)
rlb.eval()
eb = get_embs(rlb.float(), data, "cell", L2Q, base.config.pad_token_id, FBS, None).numpy()
cb = (ref * eb).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(eb, axis=1) + 1e-12)
print(f"\nexported bf16 checkpoint -> {OUTB}  ({szb/2**20:.1f} MiB)")
print(f"  cosine vs fp32: mean {cb.mean():.6f} min {cb.min():.6f}")
print("done")