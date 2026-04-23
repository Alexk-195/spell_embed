#!/usr/bin/env python3
"""Test that Python and C++ produce identical results when sharing matrices."""

import numpy as np
from word_embed import WordEmbedder, l2

# Create embedder with specific configuration
alphabet = "abcdefghijklmnopqrstuvwxyzöäüß^"
similarities = {
    'ö': [('o', 0.4), ('oe', 0.95)],
    'ä': [('a', 0.4), ('ae', 0.95)],
    'ü': [('u', 0.4), ('ue', 0.95)],
    'ß': [('ss', 0.95)]
}

embedder = WordEmbedder(alphabet, scales=(2, 3, 4), d=64, alpha=0.9, seed=42,
                        similarities=similarities)

# Save matrices to binary file
print("Saving matrices to matrices.bin...")
embedder.save_matrices('matrices.bin')
print("Matrices saved.")

# Test with a few words
test_words = [
    ("cat", "car"),
    ("guarantee", "guarentee"),
    ("könig", "koenig"),
    ("straße", "strasse"),
]

print("\nPython embeddings (with original matrices):")
for w1, w2 in test_words:
    v1 = embedder.embed(w1, normalize=True)
    v2 = embedder.embed(w2, normalize=True)
    dist = l2(v1, v2)
    print(f"  {w1:12s} vs {w2:12s}: {dist:.6f}")

# Test loading matrices back
print("\nLoading matrices back in Python...")
loaded_embedder = WordEmbedder.load_matrices('matrices.bin', alphabet, similarities=similarities)

print("\nPython embeddings (with loaded matrices):")
for w1, w2 in test_words:
    v1 = loaded_embedder.embed(w1, normalize=True)
    v2 = loaded_embedder.embed(w2, normalize=True)
    
    if w1 == "cat":
        print(f"\nDebug - cat embedding (first 5 dims): {v1[:5]}")
        print(f"Debug - car embedding (first 5 dims): {v2[:5]}")
    
    dist = l2(v1, v2)
    print(f"  {w1:12s} vs {w2:12s}: {dist:.6f}")

print("\nVerifying distances match...")
for w1, w2 in test_words:
    v1_orig = embedder.embed(w1, normalize=True)
    v2_orig = embedder.embed(w2, normalize=True)
    v1_load = loaded_embedder.embed(w1, normalize=True)
    v2_load = loaded_embedder.embed(w2, normalize=True)
    
    dist_orig = l2(v1_orig, v2_orig)
    dist_load = l2(v1_load, v2_load)
    diff = abs(dist_orig - dist_load)
    
    status = "✓" if diff < 1e-6 else "✗"
    print(f"  {status} {w1} vs {w2}: diff = {diff:.2e}")

print("\n✓ Python save/load test complete!")
print("Now run the C++ test: cd cpp && ./test_matrix_load")
