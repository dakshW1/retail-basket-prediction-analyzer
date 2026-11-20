# Retail Basket Prediction Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/vu5mcxqya7/badge.svg)](https://ask.thili.ai/b/vu5mcxqya7)

A command-line tool for retail analysts: it compares how well an **n-gram model** and a **small language model**
(distilgpt2 after continued pretraining) predict the missing item of a shopping basket. Both are trained on the same
simulated baskets, where some itemsets were planted on purpose so the right answer is known.

Track project for *Language Models from Scratch* on askThili. It builds on two drills: *Build a Next-Word Predictor
from Scratch* (the n-gram model) and *Teach a Language Model What Sells Together* (the data, the continued pretraining
and the evaluation).

## Run it

You need Python 3.12 and git. No GPU and no API keys.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch, much smaller
pip install -r requirements.txt

python -m app.cli simulate            # 6,000 baskets with 8 planted itemsets -> baskets/
python -m app.cli train               # continued pretraining of distilgpt2 -> models/basket-lm (not in git)
python -m app.cli compare             # evaluate every model -> results/comparison.md
python -m app.cli predict "Brinjal 1 kg; Spinach 500 g"   # next-item suggestions from both models
pytest                                # tests, one block per milestone (uses a 2 MB tiny GPT-2)
```

The baskets used for the results are committed in `baskets/`, so `compare` gives the same cases every time.
The trained model is about 330 MB, so it is not in git; `train` rebuilds it in about 15 minutes on a laptop CPU.

## How it works

| Milestone | What it does | Code |
|---|---|---|
| 1. Data | `thilimarketbasket.generate` builds a 150-item grocery catalogue and 6,000 ordinary baskets (items drawn by popularity). 8 itemsets of 2-4 items are planted at supports of 2-6%, chosen from the less popular half of the shop and sharing no product, so they cannot appear often by chance. A basket is a set, so every time the models see one its order is shuffled. | `simulate`, `choose_planted`, `shuffled_orders` |
| 2. n-gram | The next-word drill's model with items as words: count unigrams, bigrams and trigrams, add-k smoothing, and interpolation so an unseen context falls back to shorter ones. Probabilities sum to 1 (tested). | `NGramModel` |
| 3. Small LM | Each basket written as text (`Basket: A; B; C`) and distilgpt2 trained further on it with the plain next-token loss (padding masked, warm-up then cosine decay). To score item X the model reads the visible items and we sum log P(" X") over X's tokens, with all 150 candidates in one padded batch. | `pretrain`, `next_token_loss`, `lm_scores` |
| 4. Evaluation | For each held-out basket one item is hidden (`thilimarketbasket.evaluate.make_cases`) and each model ranks every catalogue item not already in the basket. Reported: MRR, accuracy (= hits@1), hits@3, hits@10 and mean rank, using the library's `rank_of` and `summarise`. | `evaluate` |
| 5. Comparison | A table with a 95% bootstrap interval on every MRR, plus MRR by basket size and by whether the hidden item belongs to a planted itemset. Written to `results/comparison.md` and `results/results.json`. | `breakdown`, `comparison_report` |

Two baselines frame the result: **popularity**, which never reads the basket, and **co-occurrence counts**
(the classic "bought together" counting recommender), which reads every visible item but knows nothing else.

## Results

200 held-out baskets, one item hidden in each, 150 items to rank (from `results/comparison.md`).
| Model | MRR (95% CI) | Accuracy (hits@1) | hits@3 | hits@10 | Mean rank |
|---|---|---|---|---|---|
| popularity (ignores the basket) | 0.124 (0.095-0.157) | 5.0% | 12.0% | 24.0% | 42.23 |
| 2-gram (k=1.0) | 0.115 (0.087-0.147) | 4.5% | 8.5% | 25.5% | 41.98 |
| co-occurrence counts | 0.183 (0.143-0.228) | 10.5% | 19.5% | 31.5% | 40.53 |
| small LM (distilgpt2, pretrained) | 0.106 (0.079-0.136) | 3.5% | 9.5% | 23.5% | 45.87 |

**MRR by group**

| Group | Cases | popularity (ignores the basket) | 2-gram (k=1.0) | co-occurrence counts | small LM (distilgpt2, pretrained) |
|---|---|---|---|---|---|
| basket size 2-3 | 53 | 0.123 | 0.105 | 0.115 | 0.103 |
| basket size 4-6 | 82 | 0.135 | 0.132 | 0.199 | 0.101 |
| basket size 7+ | 65 | 0.112 | 0.101 | 0.217 | 0.114 |
| hidden item is planted | 23 | 0.038 | 0.071 | 0.614 | 0.100 |
| hidden item is ordinary | 177 | 0.135 | 0.120 | 0.127 | 0.106 |

distilgpt2 continued pretraining: validation loss 5.83 -> 0.575 in 200 steps (`models/basket-lm/train_log.json`).

## What the comparison shows

- **Counting co-occurrence wins.** It is the only model clearly above popularity (MRR 0.183 vs 0.124, intervals
  barely touch), and on hidden items that belong to a planted itemset it is in a different league: MRR 0.614, against
  0.038 for popularity. When the right answer is "what was bought with these items before", counting it directly is
  hard to beat.
- **The n-gram model is no better than popularity (0.115 vs 0.124).** It sees only the last item, and in a basket the
  last item is random, so most of the basket is thrown away. On planted items it does help a little (0.071 vs 0.038),
  because when the last visible item happens to be the planted partner, the bigram has seen that pair.
- **The small LM did not beat the baselines (0.106).** Its validation loss fell a lot (5.83 -> 0.58), but most of that
  is learning the *format* and the item names (`Basket: ...; ...`), not which items go together. Its ranking ends up
  close to popularity. It does read the whole basket: on planted items it beats popularity and the n-gram
  (0.100 vs 0.038 and 0.071), which is the start of the co-occurrence signal, but 200 CPU steps were not enough to
  learn it well.
- **Basket size:** co-occurrence gets better as baskets grow (0.115 -> 0.199 -> 0.217) because more visible items
  give more evidence. The n-gram cannot use the extra items, and the LM did not yet learn to.
- **For a retail analyst:** for "bought together" suggestions on data like this, start with co-occurrence counts.
  They are cheap, explainable and the strongest here. A language model only earns its cost with much more training,
  and its real advantage (item names, text, new products) does not show on simulated data like this.
- **What I would try next:** train the LM for longer or use SmolLM2-135M as in the drill. I would also give the
  n-gram every visible item as context (which turns it into co-occurrence counting), and score only the item-name
  tokens, so the LM is not judged on how common an item's name is.

## Limitations

- The data is simulated. The planted itemsets are the only real structure, so the results show how well each model
  finds planted co-occurrence; real baskets also have seasons, households and promotions.
- 200 test cases is enough to separate popularity from the others, but close models have overlapping intervals; read
  the bootstrap ranges before calling a winner.
- distilgpt2 was trained for only 200 steps on a CPU. A longer run, or SmolLM2-135M as in the drill, would probably help.
- The n-gram model only ever sees the last one or two items. On a basket, which has no order, that throws information
  away by design; that is the point the comparison makes, not a bug.

## AI assistance

I built this with an AI coding assistant and disclosed it on the askThili ship form ("Most of it").
