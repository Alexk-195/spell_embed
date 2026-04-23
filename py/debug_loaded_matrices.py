#!/usr/bin/env python3
"""Debug: Print A matrix values after loading."""

import numpy as np
from word_embed import WordEmbedder

alphabet = "abcdefghijklmnopqrstuvwxyzöäüß^"
similarities = {
    'ö': [('o', 0.4), ('oe', 0.95)],
    'ä': [('a', 0.4), ('ae', 0.95)],
    'ü': [('u', 0.4), ('ue', 0.95)],
    'ß': [('ss', 0.95)]
}

loaded_embedder = WordEmbedder.load_matrices('matrices.bin', alphabet, similarities=similarities)

print(f"A matrix shape: {loaded_embedder.A.shape}")
print(f"A matrix first row (first 5 elements): {loaded_embedder.A[0, :5]}")
print(f"A matrix first column (first 5 elements): {loaded_embedder.A[:5, 0]}")

print(f"\nB matrices scales: {list(loaded_embedder.Bs.keys())}")
for scale in sorted(loaded_embedder.Bs.keys()):
    B = loaded_embedder.Bs[scale]
    print(f"B[{scale}] shape: {B.shape}, first 5 elements: {B[0, :5]}")
