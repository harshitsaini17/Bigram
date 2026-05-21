# Bigram

A simple **bigram language model** implemented in PyTorch that generates random Indian names by learning character-level transition probabilities from a dataset of Indian names.

## What It Does

This project trains a basic character-level bigram model on a CSV file of Indian names (`Names.csv`). It learns which characters are likely to follow each other, then uses these probabilities to generate novel, realistic-sounding Indian names.

## How It Works

1. **Data Loading** — Reads names from `Names.csv`
2. **Character Mapping** — Creates integer-to-character (`itos`) and character-to-integer (`stoi`) mappings
3. **Bigram Counting** — Builds a frequency table of character pairs
4. **Probability Distribution** — Converts counts into a probability distribution matrix
5. **Name Generation** — Samples characters sequentially based on learned bigram probabilities

### Bigram Model

```
Previous char → Next char (probability)
     .        →    a (0.12)     # Start of name
     a        →    b (0.05), n (0.31), ...
     n        →    d (0.08), a (0.22), ...
```

## Training Output

The model tracks negative log-likelihood loss during training:
```
Initial loss:     ~2.19
After training:   ~2.19 → ~2.18 (minimal, as bigram is a very simple model)
```

## Tech Stack

- **Python 3**
- **PyTorch** — tensor operations and probability sampling
- **pandas** — CSV data loading
- **NumPy** — numerical utilities
- **matplotlib** — optional visualization

## Dataset

The model is trained on a collection of Indian names stored in `Names.csv`. The dataset includes diverse names from various regions of India.

## Files

| File | Description |
|------|-------------|
| `bigram1.ipynb` | Full Jupyter notebook with data loading, model training, and name generation |
| `Names.csv` | Dataset of Indian names (referenced in notebook) |

## Usage

Open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harshitsaini17/Bigram/blob/main/bigram1.ipynb)

Or run locally:
```bash
jupyter lab bigram1.ipynb
```

## Generated Examples

The model produces names such as:
- Aabhas
- Aabhat
- Aabheer
- ...and many more unique combinations

## Learning Notes

- Bigram models are **very simple** — they only consider one previous character
- Loss converges quickly since there are limited bigram transitions
- For better results, a trigram or neural character-level RNN/Transformer would be more expressive

## Future Improvements

- [ ] Trigram model (consider 2 previous characters)
- [ ] Simple neural network (MLP-based character predictor)
- [ ] Temperature-based sampling for diversity control
- [ ] Filter generated names against a dictionary
- [ ] Add name meaning generation
