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

START = "<s>"


class NGramModel:
    """An item-level n-gram model, built the way the next-word drill builds one: count n-grams, turn counts
    into probabilities with add-k smoothing, and mix orders (interpolation) so an unseen context falls back
    to shorter ones. Each item is one "word", and a basket is a "sentence"."""

    def __init__(self, names, n: int = 3, k: float = 0.01, lambdas=(0.6, 0.3, 0.1)):
        self.names, self.n, self.k = list(names), n, k
        lam = np.array(lambdas[:n], dtype=float)[::-1]  # weights for order 0 (unigram), 1, ..., n-1
        self.weights = tuple((lam / lam.sum()).tolist())
        self.counts = [defaultdict(Counter) for _ in range(n)]  # order -> context tuple -> next item counts

    def fit(self, sequences):
        for seq in sequences:
            padded = [START] * (self.n - 1) + list(seq)
            for i in range(self.n - 1, len(padded)):
                for order in range(self.n):  # order 0 = unigram, 1 = bigram, ...
                    ctx = tuple(padded[i - order:i])
                    self.counts[order][ctx][padded[i]] += 1
        return self

    def prob(self, ctx_items, item) -> float:
        padded = [START] * (self.n - 1) + list(ctx_items)
        V = len(self.names)
        total = 0.0
        for order, lam in zip(range(self.n), self.weights):
            ctx = tuple(padded[len(padded) - order:]) if order else ()
            c = self.counts[order].get(ctx, Counter())
            total += lam * (c.get(item, 0) + self.k) / (sum(c.values()) + self.k * V)
        return total

    def scores(self, visible) -> np.ndarray:
        """log P(item | the last n-1 visible items) for every catalogue item."""
        return np.log([self.prob(visible, name) for name in self.names])


def cooccurrence_scores(train, names):
    """Item-to-item counting (the classic 'bought together' recommender): for each visible item, how
    often each other item was in the same basket. Not order-limited like the n-gram."""
    pos = {n: i for i, n in enumerate(names)}
    co = np.zeros((len(names), len(names)))
    for b in train:
        idx = [pos[x] for x in set(b)]
        for i in idx:
            co[i, idx] += 1
    np.fill_diagonal(co, 0)

    def score(visible):
        return np.log1p(sum(co[pos[v]] for v in visible)) if visible else np.zeros(len(names))
    return score


# --- Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.

def load_lm(name: str = "distilgpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tok


def to_text(items) -> str:
    from thilimarketbasket.verbalize import PREFIX, SEPARATOR

    return PREFIX + SEPARATOR.join(items)


def next_token_loss(model, ids, mask):
    """Average cross-entropy of predicting every token from the ones before it, padding masked out."""
    import torch.nn.functional as F

    logits = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1]).logits
    loss = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
    m = mask[:, 1:].float()
    return (loss * m).sum() / m.sum()


def _batch(tok, sequences):
    import torch

    enc = [tok(to_text(s))["input_ids"] + [tok.eos_token_id] for s in sequences]
    width = max(len(e) for e in enc)
    ids = torch.tensor([e + [tok.eos_token_id] * (width - len(e)) for e in enc])
    mask = torch.tensor([[1] * len(e) + [0] * (width - len(e)) for e in enc])
    return ids, mask


def validation_loss(model, tok, sequences, batch_size: int = 64) -> float:
    import torch

    model.eval()
    total = count = 0.0
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            ids, mask = _batch(tok, sequences[i:i + batch_size])
            n = float(mask[:, 1:].sum())
            total += next_token_loss(model, ids, mask).item() * n
            count += n
    return total / count


def pretrain(model, tok, sequences, val=None, steps: int = 300, batch_size: int = 16, lr: float = 1e-4,
             warmup: int = 20, eval_every: int = 50, seed: int = 0, log=print) -> list[dict]:
    """Continued pretraining with the plain next-token objective: warm-up, then cosine decay."""
    import torch

    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)
    history, start, recent = [], time.time(), []
    if val:
        history.append({"step": 0, "loss": None, "val_loss": round(validation_loss(model, tok, val), 4)})
        log(f"step 0  val loss {history[-1]['val_loss']:.3f}")
    for step in range(1, steps + 1):
        model.train()
        ids, mask = _batch(tok, rng.sample(sequences, min(batch_size, len(sequences))))
        loss = next_token_loss(model, ids, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        recent.append(loss.item())
        if step % eval_every == 0 or step == steps:
            row = {"step": step, "loss": round(float(np.mean(recent)), 4), "seconds": round(time.time() - start, 1)}
            if val:
                row["val_loss"] = round(validation_loss(model, tok, val), 4)
            history.append(row)
            recent = []
            log(f"step {step}  train {row['loss']:.3f}" + (f"  val {row['val_loss']:.3f}" if val else ""))
    model.eval()
    return history


def lm_scores(model, tok, visible, names, seed: int = 0) -> np.ndarray:
    """log P(" item" | "Basket: a; b; c;") for every catalogue item, summed over the item's tokens.
    All candidates go through the model in one padded batch."""
    import torch

    from thilimarketbasket.verbalize import verbalize_prefix

    prefix = tok(verbalize_prefix(visible, seed=seed))["input_ids"]
    cands = [tok(" " + n)["input_ids"] for n in names]
    width = len(prefix) + max(len(c) for c in cands)
    ids = torch.tensor([prefix + c + [tok.eos_token_id] * (width - len(prefix) - len(c)) for c in cands])
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits, dim=-1)
    out = np.zeros(len(names))
    for r, c in enumerate(cands):
        out[r] = sum(logp[r, len(prefix) - 1 + j, t].item() for j, t in enumerate(c))
    return out


def save_lm(model, tok, out_dir: Path = MODEL_DIR, history: list | None = None) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (Path(out_dir) / "train_log.json").write_text(json.dumps(history or [], indent=1))


# --- Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).

def make_cases(test, names, n: int = 300, seed: int = 0) -> list[dict]:
    """Hide one random item of each test basket (thilimarketbasket.evaluate.make_cases)."""
    from thilimarketbasket.evaluate import make_cases as mk

    return [c for c in mk([b for b in test if len(set(b)) >= 2], names, n, seed=seed)]


def evaluate(cases, scorer, names, ks=(1, 3, 10)) -> dict:
    """Rank every catalogue item for each case (items already in the basket are not candidates), then
    MRR, accuracy (= hits@1) and hits@K via thilimarketbasket.evaluate. Ranks are kept for the comparison."""
    from thilimarketbasket.evaluate import rank_of, summarise

    pos = {n: i for i, n in enumerate(names)}
    ranks = [rank_of(np.asarray(scorer(c["visible"]), dtype=float), pos[c["target"]],
                     [pos[v] for v in c["visible"]]) for c in cases]
    res = {k: round(v, 4) for k, v in summarise(ranks, ks=ks).items()}
    res["accuracy"] = res["hits@1"]
    res["mean_rank"] = round(float(np.mean(ranks)), 2)
    return {**res, "ranks": ranks}


def bootstrap_mrr(ranks, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the MRR: resample the test cases with replacement."""
    rng = np.random.default_rng(seed)
    rr = 1.0 / np.asarray(ranks, dtype=float)
    means = [rr[rng.integers(0, len(rr), len(rr))].mean() for _ in range(n)]
    return round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)


def popularity_scorer(train, names):
    from thilimarketbasket.evaluate import popularity_scores

    s = popularity_scores(train, names)
    return lambda visible: s


def ngram_scorer(model: NGramModel, seed: int = 0):
    """The n-gram only sees the last n-1 items, and a basket has no order, so we show it the visible
    items in a random order, exactly like the LM."""
    rng = random.Random(seed)

    def score(visible):
        v = list(visible)
        rng.shuffle(v)
        return model.scores(v)
    return score


def lm_scorer(model, tok, names, seed: int = 0):
    count = iter(range(10**9))
    return lambda visible: lm_scores(model, tok, visible, names, seed=seed + next(count))


# --- Milestone 5: Present a comparison of the models' performance.

def breakdown(cases, results: dict, planted_items: set) -> dict:
    """MRR per model for (a) basket size and (b) whether the hidden item belongs to a planted itemset."""
    groups = {"basket size 2-3": lambda c: len(c["visible"]) + 1 <= 3,
              "basket size 4-6": lambda c: 4 <= len(c["visible"]) + 1 <= 6,
              "basket size 7+": lambda c: len(c["visible"]) + 1 >= 7,
              "hidden item is planted": lambda c: c["target"] in planted_items,
              "hidden item is ordinary": lambda c: c["target"] not in planted_items}
    out = {}
    for g, test in groups.items():
        idx = [i for i, c in enumerate(cases) if test(c)]
        out[g] = {"cases": len(idx)}
        for name, r in results.items():
            out[g][name] = round(float(np.mean([1 / r["ranks"][i] for i in idx])), 4) if idx else None
    return out


def comparison_report(results: dict, groups: dict, n_cases: int) -> str:
    """A markdown summary: the main table, the breakdown, and what it means."""
    lines = [f"# Next-item prediction: n-gram vs small LM ({n_cases} held-out cases)", "",
             "| Model | MRR (95% CI) | Accuracy (hits@1) | hits@3 | hits@10 | Mean rank |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        lo, hi = bootstrap_mrr(r["ranks"])
        lines.append(f"| {name} | {r['mrr']:.3f} ({lo:.3f}-{hi:.3f}) | {r['accuracy']:.1%} | {r['hits@3']:.1%} "
                     f"| {r['hits@10']:.1%} | {r['mean_rank']} |")
    names = list(results)
    lines += ["", "## MRR by group", "", "| Group | Cases | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
    for g, row in groups.items():
        lines.append(f"| {g} | {row['cases']} | " + " | ".join(
            "-" if row[n] is None else f"{row[n]:.3f}" for n in names) + " |")
    best = max(results, key=lambda n: results[n]["mrr"])
    lines += ["", f"Best overall: **{best}**."]
    return "\n".join(lines) + "\n"


def save_results(results: dict, groups: dict, report: str) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    slim = {n: {k: v for k, v in r.items() if k != "ranks"} for n, r in results.items()}
    RESULTS_FILE.write_text(json.dumps({"results": slim, "groups": groups}, indent=1))
    REPORT_FILE.write_text(report, encoding="utf-8")
