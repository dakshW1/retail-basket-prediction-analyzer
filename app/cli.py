"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import json

from app import core


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="retail-basket-prediction-analyzer", description="Retail analysts use this to compare next-item prediction accuracy between an n-gram model and a small LM, both trained on simulated shopping baskets data.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="simulate shop baskets with planted itemsets")
    p.add_argument("--n", type=int, default=6000)
    p = sub.add_parser("train", help="continued pretraining of distilgpt2 on baskets-as-text")
    p.add_argument("--base", default="distilgpt2")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p = sub.add_parser("predict", help="likely next items for a partial basket, from both models")
    p.add_argument("items", help='items separated by ";", e.g. "Brinjal 1 kg; Spinach 500 g"')
    p.add_argument("--top", type=int, default=5)
    p = sub.add_parser("compare", help="evaluate every model on held-out baskets and write the comparison")
    p.add_argument("--cases", type=int, default=300)
    p.add_argument("--n", type=int, default=2, help="n-gram order")
    p.add_argument("--k", type=float, default=1.0, help="add-k smoothing")

    args = parser.parse_args(argv)

    if args.command == "simulate":
        baskets, planted, names = core.simulate(args.n)
        core.save_data(baskets, planted, names)
        print(f"{len(baskets)} baskets, {len(names)} items, "
              f"{sum(map(len, baskets)) / len(baskets):.1f} items per basket on average")
        for row in core.planted_report(baskets, planted):
            print(f"  planted {row['kind']}: support {row['realised']:.1%}  {' + '.join(row['items'])}")
    elif args.command == "train":
        baskets, _, _ = core.load_data()
        train, test = core.split(baskets)
        model, tok = core.load_lm(args.base)
        history = core.pretrain(model, tok, core.shuffled_orders(train, 3), val=core.shuffled_orders(test[:200], 1),
                                steps=args.steps, lr=args.lr)
        core.save_lm(model, tok, core.MODEL_DIR, history)
        print(f"saved -> {core.MODEL_DIR.relative_to(core.ROOT)}")
    elif args.command == "predict":
        baskets, _, names = core.load_data()
        train, _ = core.split(baskets)
        visible = [s.strip() for s in args.items.split(";") if s.strip()]
        unknown = [v for v in visible if v not in names]
        if unknown:
            raise SystemExit(f"not in the catalogue: {unknown}")
        ngram = core.NGramModel(names, n=2, k=1.0).fit(core.shuffled_orders(train, 3))
        model, tok = core.load_lm(str(core.MODEL_DIR))
        for label, scores in [("n-gram", ngram.scores(visible)), ("small LM", core.lm_scores(model, tok, visible, names))]:
            ranked = [names[i] for i in scores.argsort()[::-1] if names[i] not in visible][: args.top]
            print(f"{label}: " + " | ".join(ranked))
    elif args.command == "compare":
        baskets, planted, names = core.load_data()
        train, test = core.split(baskets)
        cases = core.make_cases(test, names, args.cases)
        ngram = core.NGramModel(names, n=args.n, k=args.k).fit(core.shuffled_orders(train, 3))
        model, tok = core.load_lm(str(core.MODEL_DIR))
        results = {
            "popularity (ignores the basket)": core.evaluate(cases, core.popularity_scorer(train, names), names),
            f"{args.n}-gram (k={args.k})": core.evaluate(cases, core.ngram_scorer(ngram), names),
            "co-occurrence counts": core.evaluate(cases, core.cooccurrence_scores(train, names), names),
            "small LM (distilgpt2, pretrained)": core.evaluate(cases, core.lm_scorer(model, tok, names), names),
        }
        planted_items = {i for p in planted for i in p["items"]}
        groups = core.breakdown(cases, results, planted_items)
        report = core.comparison_report(results, groups, len(cases))
        core.save_results(results, groups, report)
        print(report)
        print(json.dumps({n: {k: v for k, v in r.items() if k != "ranks"} for n, r in results.items()}))


if __name__ == "__main__":
    main()
