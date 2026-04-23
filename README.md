# Compact Character-Level Word Embedding

Training-free, deterministic word embeddings where **L2 distance correlates with edit distance**. Small misspellings or spelling variations → small distances. Unrelated words → large distances. No neural networks, no training data.

**Available in Python and C++ with full UTF-8 support.**

## How it works

Multi-scale character n-grams (bigrams, trigrams, 4-grams) are processed through a hybrid architecture:

- **Sequential pass** — sparse contractive recurrence (ESN-inspired) captures character **order**
- **Bag-of-ngrams pass** — averages projected n-grams, captures **composition**

The combination ensures anagrams are far apart while suffix variants (`orange`/`oranges`) stay close.

## Quick start

**Python:**
- Simple example: [simple_example.py](py/simple_example.py)
- Full test suite with FastText comparison: [test_suite.py](py/test_suite.py)

**C++:**
```bash
cd cpp && bash build.sh && ./test_word_embed
```

Both implementations support the same features including UTF-8 and character similarities.

## Example distances

L2 on unit-normalized embeddings (0 = identical, √2 ≈ 1.41 = orthogonal):

```
======================================================================
Compact Character-Level Word Embedding — Simple Example
======================================================================

--- Example 1: Misspellings (expect small distances) ---

  calendar        vs calender        → distance: 0.0269
  guarantee       vs guarentee       → distance: 0.0599
  separate        vs seperate        → distance: 0.1499
  receive         vs recieve         → distance: 0.1211

--- Example 2: Morphological variants (expect small distances) ---

  orange          vs oranges         → distance: 0.2325
  create          vs created         → distance: 0.1827
  create          vs creating        → distance: 0.3322

--- Example 3: Unrelated words (expect large distances) ---

  cat             vs dog             → distance: 1.1280
  apple           vs orange          → distance: 1.2027
  happy           vs river           → distance: 1.0582

--- Example 4: Anagrams & reversals (expect large distances) ---

  listen          vs silent          → distance: 0.8778
  star            vs rats            → distance: 1.1672
  dog             vs god             → distance: 0.7932

--- Example 5: German umlauts & ß (character similarity support) ---

  straße          vs strasse         → distance: 0.1211  (ß→ss)
  könig           vs konig           → distance: 0.3081  (ö→o)
  könig           vs koenig          → distance: 0.2750  (ö→oe)
  fähre           vs fahre           → distance: 0.2724  (ä→a)

--- Example 6: Country-specific spelling (US vs UK/AU English) ---

  colour          vs color           → distance: 0.3126  (-our vs -or)
  favour          vs favor           → distance: 0.3011  (-our vs -or)
  honour          vs honor           → distance: 0.2856  (-our vs -or)
  behaviour       vs behavior        → distance: 0.1249  (-our vs -or)
  centre          vs center          → distance: 0.0577  (-re vs -er)
  theatre         vs theater         → distance: 0.0383  (-re vs -er)
  metre           vs meter           → distance: 0.1377  (-re vs -er)
  organise        vs organize        → distance: 0.0251  (-ise vs -ize)
  realise         vs realize         → distance: 0.0344  (-ise vs -ize)
  recognise       vs recognize       → distance: 0.0154  (-ise vs -ize)
  travelling      vs traveling       → distance: 0.1114  (double-l inflection)
  cancelling      vs canceling       → distance: 0.1009  (double-l inflection)
  labelled        vs labeled         → distance: 0.1469  (double-l inflection)
  catalogue       vs catalog         → distance: 0.2699  (-ogue vs -og)
  dialogue        vs dialog          → distance: 0.3229  (-ogue vs -og)
  defence         vs defense         → distance: 0.0359  (-ence vs -ense)
  offence         vs offense         → distance: 0.0379  (-ence vs -ense)
  licence         vs license         → distance: 0.0325  (-ence vs -ense)
  programme       vs program         → distance: 0.3193  (-mme vs -m)
  grey            vs gray            → distance: 0.3819  (-ey vs -ay)
  jewellery       vs jewelry         → distance: 0.3207  (double-l vs single)
  mould           vs mold            → distance: 0.4988  (-ould vs -old)
  tyre            vs tire            → distance: 0.6615  (y vs i)
  aluminium       vs aluminum        → distance: 0.1261  (AU/UK vs US)

======================================================================
Summary: Separation Quality
======================================================================

  Average distance (misspellings):           0.0894
  Average distance (morphological):          0.2491
  Average distance (country-specific):      0.1958
  Average distance (unrelated/anagram):     1.0379

  Separation ratio (unrelated / similar): 5.83x

  This ratio shows how well the embedding distinguishes between
  similar word pairs (misspellings, morphology) and dissimilar ones.
  Higher is better.

```

## vs compress-fasttext
Run test_suite.py for detailed comparison. 

FastText wins on semantics (`cat` ≈ `dog`). We win on string similarity by 3x at 200x smaller size.

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d` | 64 | Embedding dimension |
| `scales` | (2,3,4) | N-gram sizes |
| `alpha` | 0.9 | Spectral radius of state matrix |
| `similarities` | None | Character similarity dict (e.g. for umlauts) |
| `length_norm` | 1.0 | 0 = length-invariant distances, 1 = normalize by word length |

## C++ Implementation

A complete C++ port is available in the `cpp/` directory with full UTF-8 support for German umlauts and other multi-byte characters.

### Building

```bash
cd cpp
bash build.sh
```

Builds:
- `test_word_embed` — Full test suite (matching Python test_suite.py)
- `test_matrix_load` — Test matrix sharing between Python and C++

### Matrix Sharing (Identical Results)

Python and C++ use different random number generators, producing different initial matrices. To get **identical embeddings** between both implementations:

```bash
# Python: Generate and save matrices
python test_matrix_sharing.py

# C++: Load same matrices
cd cpp
./test_matrix_load
```

The binary format (`matrices.bin`) contains the random projection matrices A and B, ensuring bit-for-bit identical embeddings across languages.

**Verified identical results:**
```
cat vs car:            0.416670  ✓
guarantee vs guarentee: 0.034392  ✓
könig vs koenig:       0.249254  ✓
straße vs strasse:     0.103503  ✓
```

### Usage (C++)

```cpp
#include "word_embed.h"

// Create embedder
WordEmbedConfig config;
config.alphabet = "abcdefghijklmnopqrstuvwxyzöäüß^";
config.dimensions = 64;
config.similarities = {
    {"ö", {{"o", 0.4}, {"oe", 0.95}}},
    {"ä", {{"a", 0.4}, {"ae", 0.95}}},
    {"ü", {{"u", 0.4}, {"ue", 0.95}}},
    {"ß", {{"ss", 0.95}}}
};

auto embedder = WordEmbedder::create(config);

// Embed words
std::vector<float> v1, v2;
embedder->embed("König", v1);
embedder->embed("Koenig", v2);

// Or load from shared matrices
auto embedder2 = WordEmbedder::create_from_file("matrices.bin", config);
```

### UTF-8 Support

The C++ implementation is fully UTF-8 aware:
- Alphabet and n-grams operate on UTF-8 characters (not bytes)
- Similarities map uses `std::string` keys for multi-byte characters
- Correctly handles German umlauts: ö (2 bytes), ä (2 bytes), ü (2 bytes), ß (2 bytes)

## Dependencies

**Python:** NumPy only.

**C++:** C++17 standard library only (no external dependencies).

## Details

See [word_embed.md](word_embed.md) for the full algorithm description, design rationale, and empirical evaluation.