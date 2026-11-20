"""Tests: one block per milestone in README.md. LM tests use `sshleifer/tiny-gpt2` (~2 MB) to stay fast."""
import numpy as np
import pytest

from app import core

TINY = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def shop():
    baskets, planted, names = core.simulate(1500, seed=1)
    train, test = core.split(baskets)
    return baskets, planted, names, train, test


# Milestone 1: Generate a synthetic dataset of shopping baskets with planted itemsets.
def test_planted_itemsets_hit_their_support(shop):
    baskets, planted, _, _, _ = shop
    for row in core.planted_report(baskets, planted):
        assert row["realised"] == pytest.approx(row["target"], abs=0.005)


def test_planted_itemsets_share_no_items(shop):
    _, planted, names, _, _ = shop
    items = [i for p in planted for i in p.items]
    assert len(items) == len(set(items)) and set(items) <= set(names)


def test_baskets_are_sets_and_data_round_trips(shop, tmp_path):
    baskets, planted, names, _, _ = shop
    assert all(len(b) == len(set(b)) >= 1 for b in baskets)
    core.save_data(baskets[:10], planted, names, tmp_path)
    b2, p2, n2 = core.load_data(tmp_path)
    assert b2 == baskets[:10] and n2 == names and p2[0]["items"] == list(planted[0].items)


def test_shuffled_orders_keep_the_items(shop):
    baskets = shop[0][:20]
    orders = core.shuffled_orders(baskets, epochs=2)
    assert len(orders) == 40 and all(sorted(o) == sorted(b) for o, b in zip(orders, baskets * 2))


# Milestone 2: Implement the n-gram model to predict next items in a basket.
def test_ngram_probabilities_sum_to_one(shop):
    _, _, names, train, _ = shop
    for n in (1, 2, 3):
        m = core.NGramModel(names, n=n, k=0.5).fit(core.shuffled_orders(train[:300], 1))
        assert sum(m.prob([train[0][0]], x) for x in names) == pytest.approx(1.0)


def test_bigram_learns_a_planted_pair():
    names = ["a", "b", "c", "d"]
    m = core.NGramModel(names, n=2, k=0.01, lambdas=(0.9, 0.1)).fit([["a", "b"], ["b", "a"]] * 50 + [["c", "d"]])
    scores = m.scores(["a"])
    assert names[int(np.argmax(scores))] == "b"


# Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
@pytest.fixture(scope="module")
def lm(shop):
    _, _, _, train, _ = shop
    model, tok = core.load_lm(TINY)
    seqs = core.shuffled_orders(train[:60], 1)
    history = core.pretrain(model, tok, seqs, val=seqs[:20], steps=20, batch_size=8, lr=5e-2,
                            warmup=2, eval_every=10, log=lambda *_: None)
    return model, tok, history


def test_continued_pretraining_lowers_validation_loss(lm):
    assert lm[2][-1]["val_loss"] < lm[2][0]["val_loss"]


def test_lm_scores_every_item(lm, shop):
    model, tok, _ = lm
    names = shop[2]
    s = core.lm_scores(model, tok, [names[0], names[1]], names)
    assert s.shape == (len(names),) and np.all(s < 0)


# Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
def test_cases_hide_one_real_item(shop):
    _, _, names, _, test = shop
    for c in core.make_cases(test, names, 40):
        assert c["target"] not in c["visible"] and any(set(c["visible"]) | {c["target"]} == set(b) for b in test)


def test_evaluate_metrics_and_baseline_order(shop):
    _, _, names, train, test = shop
    cases = core.make_cases(test, names, 150)
    pop = core.evaluate(cases, core.popularity_scorer(train, names), names)
    co = core.evaluate(cases, core.cooccurrence_scores(train, names), names)
    assert {"mrr", "accuracy", "hits@3", "hits@10", "mean_rank"} <= set(pop)
    assert pop["accuracy"] == pop["hits@1"] and 0 < pop["mrr"] <= 1
    assert co["mrr"] > pop["mrr"]  # reading the basket beats ignoring it


def test_lm_runs_in_evaluate(lm, shop):
    model, tok, _ = lm
    _, _, names, _, test = shop
    res = core.evaluate(core.make_cases(test, names, 5), core.lm_scorer(model, tok, names), names)
    assert 0 < res["mrr"] <= 1 and len(res["ranks"]) == 5


# Milestone 5: Present a comparison of the models' performance.
def test_comparison_report_and_breakdown(shop):
    _, planted, names, train, test = shop
    cases = core.make_cases(test, names, 80)
    results = {"popularity": core.evaluate(cases, core.popularity_scorer(train, names), names),
               "co-occurrence": core.evaluate(cases, core.cooccurrence_scores(train, names), names)}
    groups = core.breakdown(cases, results, {i for p in planted for i in p.items})
    assert sum(groups[g]["cases"] for g in ("basket size 2-3", "basket size 4-6", "basket size 7+")) == len(cases)
    assert groups["hidden item is planted"]["cases"] + groups["hidden item is ordinary"]["cases"] == len(cases)
    report = core.comparison_report(results, groups, len(cases))
    assert "| popularity |" in report and "MRR by group" in report


def test_bootstrap_interval_contains_the_mrr():
    ranks = [1, 2, 5, 1, 10, 3, 1, 4]
    lo, hi = core.bootstrap_mrr(ranks)
    assert lo <= np.mean(1 / np.array(ranks)) <= hi
