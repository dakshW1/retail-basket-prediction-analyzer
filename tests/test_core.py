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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Present a comparison of the models' performance.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

