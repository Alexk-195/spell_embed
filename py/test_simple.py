#!/usr/bin/env python3
"""Test with English words only (no German special chars)."""

import numpy as np
from word_embed import WordEmbedder, l2

# Create embedder WITHOUT German similarities
alphabet = "abcdefghijklmnopqrstuvwxyz^"
embedder = WordEmbedder(alphabet, scales=(2, 3, 4), d=64, alpha=0.9, seed=42,
                        similarities=None)  # No similarities

# Save matrices
print("Saving matrices (no German chars) to matrices_simple.bin...")
embedder.save_matrices('matrices_simple.bin')

# Test words (English only)
test_words = [
    ("cat", "car"),
    ("guarantee", "guarentee"),
    ("hello", "helo"),
]

print("\nPython embeddings (original):")
for w1, w2 in test_words:
    v1 = embedder.embed(w1, normalize=True)
    v2 = embedder.embed(w2, normalize=True)
    if w1 == "cat":
        print(f"  Debug - cat: {v1[:5]}")
        print(f"  Debug - car: {v2[:5]}")
    dist = l2(v1, v2)
    print(f"  {w1:12s} vs {w2:12s}: {dist:.6f}")

# Load and test
loaded = WordEmbedder.load_matrices('matrices_simple.bin', alphabet, similarities=None)
print("\nPython embeddings (loaded):")
for w1, w2 in test_words:
    v1 = loaded.embed(w1, normalize=True)
    v2 = loaded.embed(w2, normalize=True)
    dist = l2(v1, v2)
    print(f"  {w1:12s} vs {w2:12s}: {dist:.6f}")
