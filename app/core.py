"""Retail Basket Prediction Analyzer: n-gram model vs a small LM on next-item prediction.

  1. data       - simulated shop baskets with itemsets planted on purpose (thilimarketbasket.generate)
  2. n-gram     - bigram/trigram counts over basket text with add-k smoothing and interpolation
  3. small LM   - continued pretraining of distilgpt2 on baskets written as text, item order shuffled
  4. evaluation - hide one item per test basket, rank every catalogue item: MRR, accuracy (hits@1), hits@K
  5. comparison - side-by-side table, a breakdown by basket size and by planted vs ordinary items
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "baskets"
BASKETS_FILE = DATA_DIR / "baskets.jsonl"
PLANTED_FILE = DATA_DIR / "planted.json"
MODEL_DIR = ROOT / "models" / "basket-lm"
RESULTS_FILE = ROOT / "results" / "results.json"
REPORT_FILE = ROOT / "results" / "comparison.md"


# --- Milestone 1: Generate a synthetic dataset of shopping baskets with planted itemsets.

# (size of the itemset, support) for each itemset we plant: the answer key the models are scored against.
PLANT_PLAN = [(2, 0.06), (3, 0.05), (2, 0.04), (4, 0.04), (3, 0.03), (2, 0.03), (3, 0.02), (2, 0.02)]


def make_shop(n_items: int = 150, seed: int = 0):
    """The catalogue and each item's share of sales, from thilimarketbasket.generate."""
    from thilimarketbasket import generate as g

    catalog = g.make_catalog(n_items=n_items, seed=seed)
    return catalog, g.popularity(catalog)


def choose_planted(catalog, shares, seed: int = 0, plan=PLANT_PLAN):
    """Pick itemsets that share no item and no product, from the less popular half of the shop,
    so they cannot happen often by chance: any strong co-occurrence the models find was planted."""
    from thilimarketbasket.generate import Planted

    rng = random.Random(seed)
    order = list(np.argsort(shares))[: len(catalog) // 2]  # the less popular half
    rng.shuffle(order)
    used_bases, planted = set(), []
    for size, support in plan:
        picked = []
        for i in order:
            item = catalog[i]
            if item.base not in used_bases:
                picked.append(item.name)
                used_bases.add(item.base)
            if len(picked) == size:
                break
        order = [i for i in order if catalog[i].name not in picked]
        planted.append(Planted(items=tuple(picked), support=support, kind=f"size {size}"))
    return planted


def simulate(n_baskets: int = 6000, n_items: int = 150, seed: int = 0, mean_size: float = 5.0):
    """Background baskets (items drawn by popularity) with the planted itemsets added on top."""
    from thilimarketbasket import generate as g

    catalog, shares = make_shop(n_items, seed)
    background = g.background_baskets(n_baskets, catalog, shares, seed=seed, mean_size=mean_size)
    planted = choose_planted(catalog, shares, seed)
    baskets = g.plant_all(background, planted, catalog, seed=seed)
    baskets = [sorted(set(b)) for b in baskets]  # a basket is a set: quantities are dropped
    names = sorted({i.name for i in catalog})
    return baskets, planted, names


def planted_report(baskets, planted) -> list[dict]:
    from thilimarketbasket.generate import planted_report as report

    return report(baskets, planted)


def save_data(baskets, planted, names, data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "baskets.jsonl").write_text("".join(json.dumps(b) + "\n" for b in baskets), encoding="utf-8")
    info = {"names": names, "planted": [{"items": list(p.items), "support": p.support} for p in planted]}
    (data_dir / "planted.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


def load_data(data_dir: Path = DATA_DIR):
    baskets = [json.loads(l) for l in (data_dir / "baskets.jsonl").read_text(encoding="utf-8").splitlines() if l]
    info = json.loads((data_dir / "planted.json").read_text(encoding="utf-8"))
    return baskets, info["planted"], info["names"]


def split(baskets, test_share: float = 0.15):
    cut = int(len(baskets) * (1 - test_share))
    return baskets[:cut], baskets[cut:]


def shuffled_orders(baskets, epochs: int = 3, seed: int = 0) -> list[list[str]]:
    """Each basket in `epochs` fresh random orders. A basket has no order, so any fixed order would be
    something the models learn that means nothing (the drill's lesson on verbalizing)."""
    rng = random.Random(seed)
    out = []
    for _ in range(epochs):
        for b in baskets:
            b = list(b)
            rng.shuffle(b)
            out.append(b)
    return out


# --- Milestone 2: Implement the n-gram model to predict next items in a basket.
# Hint: Use the n-gram model construction method from 'Build a Next-Word Predictor from Scratch'.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
# Hint: Train your small language model following the approach in 'Teach a Language Model What Sells Together'.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
# Hint: Calculate next-item prediction metrics such as accuracy and rank for both models.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Present a comparison of the models' performance.
# Hint: Summarize and compare the evaluation results to highlight model strengths and weaknesses.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
