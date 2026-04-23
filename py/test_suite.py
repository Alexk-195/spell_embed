#!/usr/bin/env python3
import numpy as np
from word_embed import WordEmbedder, l2

# =============================================================================
# Test suite
# =============================================================================

if __name__ == "__main__":
    # Alphabet: 26 lowercase letters + German special chars + start boundary
    alphabet = "abcdefghijklmnopqrstuvwxyzöäüß^"

    # Character similarity mappings for German.
    # Single-char targets → soft encoding (ö looks like o in the encoding).
    # Multi-char targets  → expansion blending (embed "König" blended with "Koenig").
    similarities = {
        'ö': [('o', 0.4), ('oe', 0.95)],
        'ä': [('a', 0.4), ('ae', 0.95)],
        'ü': [('u', 0.4), ('ue', 0.95)],
        'ß': [('ss', 0.95)]
    }

    dimensions = 64
    seed = 42
    embedder = WordEmbedder(alphabet, scales=(2, 3, 4), d=dimensions, alpha=0.9, seed=seed,
                            similarities=similarities)

    embedder.save_matrices('../bin/matrices.bin')

    # -------------------------------------------------------------------------
    # Test groups — each group isolates a specific embedding property.
    #
    # Expected behavior:
    #   - Misspellings:        SMALL distance  (edit distance 1-2)
    #   - Morphological:       SMALL distance  (shared root)
    #   - Position symmetry:   SIMILAR distances regardless of edit position
    #   - Unrelated:           LARGE distance
    #   - Anagram / reversal:  LARGE distance  (same letters, different order)
    # -------------------------------------------------------------------------

    groups = {
        "Misspellings (expect: small distances)": [
            # Single character deletions
            ("guarantee", "guarante",   "1 deletion (middle)"),
            ("accept",    "acept",      "1 deletion (middle)"),
            ("library",   "libary",     "1 deletion (middle)"),
            ("calendar",  "calender",   "1 substitution (middle)"),
            # Substitutions
            ("guarantee", "guarentee",  "1 substitution"),
            ("separate",  "seperate",   "1 substitution (common mistake)"),
            ("definite",  "definate",   "1 substitution (common mistake)"),
            ("receive",   "recieve",    "transposition (ie→ei)"),
            ("necessary", "neccessary", "1 insertion (double c)"),
            ("necessary", "necesary",   "1 deletion"),
            # Start/end edits
            ("accept",    "except",     "1 substitution at start"),
            ("knight",    "night",      "1 deletion at start"),
            ("their",     "there",      "1 substitution + deletion at end"),
            ("16-bit",    "16 bit",     "1 substitution with letters both not in alphabet(treated as same)"),
            ("16-bit",    "16bit",      "1 insertion of letter not in alphabet"),
        ],
        "Morphological (expect: small distances)": [
            ("orange",    "oranges",    "plural suffix -s"),
            ("run",       "running",    "verb form -ning"),
            ("happy",     "happily",    "adverb form -ily"),
            ("create",    "created",    "past tense -d"),
            ("create",    "creating",   "gerund -ing"),
            ("quick",     "quickly",    "adverb -ly"),
        ],
        "Position symmetry (expect: similar distances for all)": [
            # All pairs have exactly 1 character substitution
            ("cat",  "bat",  "edit at start (c→b)"),
            ("cat",  "car",  "edit at end (t→r)"),
            ("cat",  "cot",  "edit in middle (a→o)"),
            ("lamp", "ramp", "edit at start (l→r)"),
            ("lamp", "lump", "edit in middle (a→u)"),
            ("lamp", "lamb", "edit at end (p→b)"),
        ],
        "Unrelated words (expect: large distances)": [
            ("guarantee", "banana",     "long vs medium"),
            ("cat",       "banana",     "short vs medium"),
            ("apple",     "orange",     "same length"),
            ("cat",       "dog",        "short vs short"),
            ("elephant",  "telephone",  "shared letters, unrelated"),
            ("happy",     "river",      "same length, no overlap"),
        ],
        "Anagram / reversal (expect: large distances)": [
            # Same characters, different order — should be far apart
            ("listen",  "silent",   "anagram"),
            ("star",    "rats",     "reversal"),
            ("dog",     "god",      "reversal"),
            ("space",   "capes",    "anagram"),
        ],
        "German umlaut→base (expect: small distances after similarity)": [
            # ö→o, ä→a, ü→u — single char substitution but currently
            # ö and o are unrelated alphabet entries, so distances will be large
            ("könig",    "konig",    "ö→o"),
            ("schön",    "schon",    "ö→o"),
            ("größe",    "grosse",   "ö→o + ß→ss (length change)"),
            ("über",     "uber",     "ü→u"),
            ("müller",   "muller",   "ü→u"),
            ("fähre",    "fahre",    "ä→a"),
            ("bär",      "bar",      "ä→a"),
            ("hände",    "hande",    "ä→a"),
        ],
        "German umlaut→transcription (expect: small distances after similarity)": [
            # ö→oe, ä→ae, ü→ue — length changes, currently very different
            ("könig",    "koenig",   "ö→oe"),
            ("schön",    "schoen",   "ö→oe"),
            ("über",     "ueber",    "ü→ue"),
            ("müller",   "mueller",  "ü→ue"),
            ("fähre",    "faehre",   "ä→ae"),
            ("bär",      "baer",     "ä→ae"),
            ("hände",    "haende",   "ä→ae"),
        ],
        "German ß variants (expect: small distances after similarity)": [
            # ß→ss — length change
            ("straße",   "strasse",  "ß→ss"),
            ("groß",     "gross",    "ß→ss"),
            ("fuß",      "fuss",     "ß→ss"),
            ("maß",      "mass",     "ß→ss"),
            ("heißen",   "heissen",  "ß→ss"),
            ("blumenstrauß", "blumenstrauss", "ß→ss"),
            ("blumenstraße", "blumenstrasse", "ß→ss"),            
        ],
        "German unrelated (expect: large distances)": [
            # Sanity check: unrelated German words should stay far apart
            ("könig",    "straße",   "unrelated"),
            ("über",     "hände",    "unrelated"),
            ("schön",    "müller",   "unrelated"),
            ("bär",      "fuß",      "unrelated"),
            ("strauß", "straße", "unrelated"),                   
        ],
        "Country-specific spelling (US vs UK/AU English) (expect: small distances)": [
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
        ],
    }

    # -------------------------------------------------------------------------
    # Load compress-fasttext model for comparison
    # -------------------------------------------------------------------------

    ft_model = None
    try:
        import compress_fasttext
        print("Loading compress-fasttext English model (~20MB)...")
        ft_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
            '../bin/cc.en.300.compressed.bin'
        )
        print("Loaded.\n")
    except Exception as e:
        print(f"Could not load compress-fasttext: {e}")
        print("Running without FastText comparison.\n")

    # -------------------------------------------------------------------------
    # Embed all unique words with both models
    # -------------------------------------------------------------------------

    all_words = set()
    for pairs in groups.values():
        for w1, w2, _ in pairs:
            all_words.add(w1)
            all_words.add(w2)

    # Our embeddings (normalized for fair L2 comparison)
    vectors_ours = {w: embedder.embed(w, normalize=True) for w in all_words}

    # FastText embeddings (also normalized)
    vectors_ft = {}
    if ft_model is not None:
        for w in all_words:
            v = ft_model[w]
            norm = np.linalg.norm(v)
            vectors_ft[w] = v / norm if norm > 0 else v

    # -------------------------------------------------------------------------
    # Print side-by-side comparison
    # -------------------------------------------------------------------------

    header_ours = "Ours"
    header_ft = "FastText"

    for group_name, pairs in groups.items():
        print(f"--- {group_name} ---")
        if ft_model is not None:
            print(f"  {'':12s}    {'':12s}   {header_ours:>8s}  {header_ft:>8s}")
        for w1, w2, desc in pairs:
            d_ours = l2(vectors_ours[w1], vectors_ours[w2])
            line = f"  {w1:12s} vs {w2:12s}: {d_ours:8.4f}"
            if ft_model is not None:
                d_ft = l2(vectors_ft[w1], vectors_ft[w2])
                line += f"  {d_ft:8.4f}"
            line += f"   ({desc})"
            print(line)
        print()

    # -------------------------------------------------------------------------
    # Summary statistics: separation quality
    # -------------------------------------------------------------------------

    # Collect distances by category
    def avg_dist(pairs_list, vectors):
        return np.mean([l2(vectors[w1], vectors[w2]) for w1, w2, _ in pairs_list])

    similar_groups = ["Misspellings (expect: small distances)",
                      "Morphological (expect: small distances)",
                      "Country-specific spelling (US vs UK/AU English) (expect: small distances)"]
    distant_groups = ["Unrelated words (expect: large distances)",
                      "Anagram / reversal (expect: large distances)"]
    german_similar_groups = ["German umlaut→base (expect: small distances after similarity)",
                             "German umlaut→transcription (expect: small distances after similarity)",
                             "German ß variants (expect: small distances after similarity)"]
    german_distant_groups = ["German unrelated (expect: large distances)"]

    similar_pairs = []
    distant_pairs = []
    for name in similar_groups:
        similar_pairs.extend(groups[name])
    for name in distant_groups:
        distant_pairs.extend(groups[name])

    german_similar_pairs = []
    german_distant_pairs = []
    for name in german_similar_groups:
        german_similar_pairs.extend(groups[name])
    for name in german_distant_groups:
        german_distant_pairs.extend(groups[name])

    print("=" * 65)
    print("SUMMARY: Separation quality (higher = better discrimination)")
    print("=" * 65)
    print()

    avg_sim_ours = avg_dist(similar_pairs, vectors_ours)
    avg_far_ours = avg_dist(distant_pairs, vectors_ours)
    ratio_ours = avg_far_ours / avg_sim_ours

    print(f"  {'Metric':<35s} {header_ours:>8s}", end="")
    if ft_model is not None:
        print(f"  {header_ft:>8s}", end="")
    print()
    print(f"  {'-'*35} {'-'*8}", end="")
    if ft_model is not None:
        print(f"  {'-'*8}", end="")
    print()

    print(f"  {'Avg similar distance':<35s} {avg_sim_ours:8.4f}", end="")
    if ft_model is not None:
        avg_sim_ft = avg_dist(similar_pairs, vectors_ft)
        print(f"  {avg_sim_ft:8.4f}", end="")
    print()

    print(f"  {'Avg unrelated distance':<35s} {avg_far_ours:8.4f}", end="")
    if ft_model is not None:
        avg_far_ft = avg_dist(distant_pairs, vectors_ft)
        print(f"  {avg_far_ft:8.4f}", end="")
    print()

    print(f"  {'Separation ratio (far/similar)':<35s} {ratio_ours:8.2f}x", end="")
    if ft_model is not None:
        ratio_ft = avg_far_ft / avg_sim_ft
        print(f"  {ratio_ft:8.2f}x", end="")
    print()

    print()
    print(f"  {'Embedding dimension':<35s} {str(dimensions):>8s}", end="")
    if ft_model is not None:
        print(f"  {'300':>8s}", end="")
    print()

    print(f"  {'Model size (approx)':<35s} {'~100KB':>8s}", end="")
    if ft_model is not None:
        print(f"  {'~20MB':>8s}", end="")
    print()

    print(f"  {'Requires training data':<35s} {'No':>8s}", end="")
    if ft_model is not None:
        print(f"  {'Yes':>8s}", end="")
    print()

    print(f"  {'Handles any string':<35s} {'Yes':>8s}", end="")
    if ft_model is not None:
        print(f"  {'Yes*':>8s}", end="")
    print()
    if ft_model is not None:
        print("  * FastText handles OOV via subword hashing")
    print()

    # -------------------------------------------------------------------------
    # German-specific separation quality
    # -------------------------------------------------------------------------

    avg_gsim_ours = avg_dist(german_similar_pairs, vectors_ours)
    avg_gfar_ours = avg_dist(german_distant_pairs, vectors_ours)
    ratio_g_ours = avg_gfar_ours / avg_gsim_ours if avg_gsim_ours > 0 else float('inf')

    print("=" * 65)
    print("GERMAN: Separation quality (umlaut/ß variants vs unrelated)")
    print("=" * 65)
    print()
    print(f"  {'Avg similar distance (ö→o,oe etc)':<40s} {avg_gsim_ours:8.4f}")
    print(f"  {'Avg unrelated distance':<40s} {avg_gfar_ours:8.4f}")
    print(f"  {'Separation ratio (far/similar)':<40s} {ratio_g_ours:8.2f}x")
    print()

    # -------------------------------------------------------------------------
    # Length normalization comparison
    #
    # Same edit type (1 substitution in middle) across different word lengths.
    # Shows how length_norm controls whether long words have smaller per-edit
    # distances (length_norm=1.0) or similar per-edit distances (length_norm=0.0).
    # -------------------------------------------------------------------------

    length_test_pairs = [
        ("cat",       "cot",       "3 chars"),
        ("lamp",      "lump",      "4 chars"),
        ("stone",     "stone",     "5 chars"),   # baseline: same word
        ("happy",     "hippy",     "5 chars"),
        ("bridge",    "briage",    "6 chars"),
        ("calendar",  "calender",  "8 chars"),
        ("separate",  "seperate",  "8 chars"),
        ("guarantee", "guarentee", "9 chars"),
        ("committee", "comittee",  "9 chars"),
    ]

    ln_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("=" * 78)
    print("LENGTH NORMALIZATION: same edit type, varying word length")
    print("  length_norm=1.0 (default): long words → smaller distances")
    print("  length_norm=0.0:           distances independent of word length")
    print("=" * 78)
    print()

    # Header
    header = f"  {'Pair':<28s}"
    for ln in ln_values:
        header += f"  ln={ln:<4.2f}"
    print(header)
    print(f"  {'-'*28}" + f"  {'-'*8}" * len(ln_values))

    # Build embedders for each length_norm value (reuse same seed + params)
    ln_embedders = {}
    for ln in ln_values:
        ln_embedders[ln] = WordEmbedder(alphabet, scales=(2, 3, 4), d=dimensions, alpha=0.9,
                                        seed=seed, similarities=similarities, length_norm=ln)

    for w1, w2, desc in length_test_pairs:
        if w1 == w2:
            continue
        line = f"  {w1:>12s}/{w2:<14s}"
        for ln in ln_values:
            e = ln_embedders[ln]
            v1 = e.embed(w1, normalize=True)
            v2 = e.embed(w2, normalize=True)
            d = l2(v1, v2)
            line += f"  {d:8.4f}"
        line += f"   ({desc})"
        print(line)
    print()

