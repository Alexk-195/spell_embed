#!/usr/bin/env python3
"""
Simple example demonstrating the compact character-level word embedding.

This example shows:
  - Basic usage with English words
  - Misspellings and morphological variants
  - German umlaut support via character similarities
  - Comparison of distances across different word pair categories
"""

import numpy as np
from py.word_embed import WordEmbedder, l2


def main():
    # =========================================================================
    # Setup: alphabet and German character similarities
    # =========================================================================
    
    alphabet = "abcdefghijklmnopqrstuvwxyz^öäüß"
    
    # Character similarity mappings for German umlauts and ß
    similarities = {
        'ö': [('o', 0.7), ('oe', 0.5)],
        'ä': [('a', 0.7), ('ae', 0.5)],
        'ü': [('u', 0.7), ('ue', 0.5)],
        'ß': [('ss', 0.7)],
    }
    
    # Create embedder with default parameters
    embedder = WordEmbedder(alphabet, similarities=similarities)
    
    print("=" * 70)
    print("Compact Character-Level Word Embedding — Simple Example")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Example 1: Misspellings (small distances)
    # =========================================================================
    
    print("--- Example 1: Misspellings (expect small distances) ---")
    print()
    
    misspelling_pairs = [
        ("calendar", "calender"),
        ("guarantee", "guarentee"),
        ("separate", "seperate"),
        ("receive", "recieve"),
    ]
    
    for word1, word2 in misspelling_pairs:
        v1 = embedder.embed(word1, normalize=True)
        v2 = embedder.embed(word2, normalize=True)
        distance = l2(v1, v2)
        print(f"  {word1:15s} vs {word2:15s} → distance: {distance:.4f}")
    
    print()
    
    # =========================================================================
    # Example 2: Morphological variants (small distances)
    # =========================================================================
    
    print("--- Example 2: Morphological variants (expect small distances) ---")
    print()
    
    morphological_pairs = [
        ("orange", "oranges"),
        ("create", "created"),
        ("create", "creating"),
    ]
    
    for word1, word2 in morphological_pairs:
        v1 = embedder.embed(word1, normalize=True)
        v2 = embedder.embed(word2, normalize=True)
        distance = l2(v1, v2)
        print(f"  {word1:15s} vs {word2:15s} → distance: {distance:.4f}")
    
    print()
    
    # =========================================================================
    # Example 3: Unrelated words (large distances)
    # =========================================================================
    
    print("--- Example 3: Unrelated words (expect large distances) ---")
    print()
    
    unrelated_pairs = [
        ("cat", "dog"),
        ("apple", "orange"),
        ("happy", "river"),
    ]
    
    for word1, word2 in unrelated_pairs:
        v1 = embedder.embed(word1, normalize=True)
        v2 = embedder.embed(word2, normalize=True)
        distance = l2(v1, v2)
        print(f"  {word1:15s} vs {word2:15s} → distance: {distance:.4f}")
    
    print()
    
    # =========================================================================
    # Example 4: Anagrams / reversals (large distances despite same letters)
    # =========================================================================
    
    print("--- Example 4: Anagrams & reversals (expect large distances) ---")
    print()
    
    anagram_pairs = [
        ("listen", "silent"),
        ("star", "rats"),
        ("dog", "god"),
    ]
    
    for word1, word2 in anagram_pairs:
        v1 = embedder.embed(word1, normalize=True)
        v2 = embedder.embed(word2, normalize=True)
        distance = l2(v1, v2)
        print(f"  {word1:15s} vs {word2:15s} → distance: {distance:.4f}")
    
    print()
    
    # =========================================================================
    # Example 5: German character similarities
    # =========================================================================
    
    print("--- Example 5: German umlauts & ß (character similarity support) ---")
    print()
    
    german_pairs = [
        ("straße", "strasse", "ß→ss"),
        ("könig", "konig", "ö→o"),
        ("könig", "koenig", "ö→oe"),
        ("fähre", "fahre", "ä→a"),
    ]
    
    for word1, word2, desc in german_pairs:
        v1 = embedder.embed(word1, normalize=True)
        v2 = embedder.embed(word2, normalize=True)
        distance = l2(v1, v2)
        print(f"  {word1:15s} vs {word2:15s} → distance: {distance:.4f}  ({desc})")
    
    print()
    
    # =========================================================================
    # Example 6: Country-specific spelling differences (US vs UK/AU)
    # =========================================================================
    
    print("--- Example 6: Country-specific spelling (US vs UK/AU English) ---")
    print()
    
    country_spelling_pairs = [
        # -our vs -or
        ("colour", "color", "-our vs -or"),
        ("favour", "favor", "-our vs -or"),
        ("honour", "honor", "-our vs -or"),
        ("behaviour", "behavior", "-our vs -or"),
        # -re vs -er
        ("centre", "center", "-re vs -er"),
        ("theatre", "theater", "-re vs -er"),
        ("metre", "meter", "-re vs -er"),
        # -ise vs -ize
        ("organise", "organize", "-ise vs -ize"),
        ("realise", "realize", "-ise vs -ize"),
        ("recognise", "recognize", "-ise vs -ize"),
        # Double-L differences
        ("travelling", "traveling", "double-l inflection"),
        ("cancelling", "canceling", "double-l inflection"),
        ("labelled", "labeled", "double-l inflection"),
        # -ogue vs -og
        ("catalogue", "catalog", "-ogue vs -og"),
        ("dialogue", "dialog", "-ogue vs -og"),
        # Miscellaneous
        ("defence", "defense", "-ence vs -ense"),
        ("offence", "offense", "-ence vs -ense"),
        ("licence", "license", "-ence vs -ense"),
        ("programme", "program", "-mme vs -m"),
        ("grey", "gray", "-ey vs -ay"),
        ("jewellery", "jewelry", "double-l vs single"),
        ("mould", "mold", "-ould vs -old"),
        ("tyre", "tire", "y vs i"),
        # AU-specific
        ("aluminium", "aluminum", "AU/UK vs US"),
    ]
    
    for word1, word2, desc in country_spelling_pairs:
        v1 = embedder.embed(word1, normalize=True)
        v2 = embedder.embed(word2, normalize=True)
        distance = l2(v1, v2)
        print(f"  {word1:15s} vs {word2:15s} → distance: {distance:.4f}  ({desc})")
    
    print()
    
    # =========================================================================
    # Summary: Separation quality
    # =========================================================================
    
    print("=" * 70)
    print("Summary: Separation Quality")
    print("=" * 70)
    print()
    
    # Compute average distances for different categories
    all_misspellings = misspelling_pairs
    all_morphological = morphological_pairs
    all_country_spelling = country_spelling_pairs
    all_unrelated = unrelated_pairs + anagram_pairs
    
    def avg_distance(pairs):
        distances = []
        for pair in pairs:
            w1, w2 = pair[:2]  # Handle both 2-tuples and 3-tuples
            v1 = embedder.embed(w1, normalize=True)
            v2 = embedder.embed(w2, normalize=True)
            distances.append(l2(v1, v2))
        return np.mean(distances)
    
    avg_misspell = avg_distance(all_misspellings)
    avg_morph = avg_distance(all_morphological)
    avg_country = avg_distance(all_country_spelling)
    avg_unrel = avg_distance(all_unrelated)
    
    avg_similar = (avg_misspell + avg_morph + avg_country) / 3
    separation_ratio = avg_unrel / avg_similar
    
    print(f"  Average distance (misspellings):           {avg_misspell:.4f}")
    print(f"  Average distance (morphological):          {avg_morph:.4f}")
    print(f"  Average distance (country-specific):      {avg_country:.4f}")
    print(f"  Average distance (unrelated/anagram):     {avg_unrel:.4f}")
    print()
    print(f"  Separation ratio (unrelated / similar): {separation_ratio:.2f}x")
    print()
    print("  This ratio shows how well the embedding distinguishes between")
    print("  similar word pairs (misspellings, morphology) and dissimilar ones.")
    print("  Higher is better.")
    print()


if __name__ == "__main__":
    main()
