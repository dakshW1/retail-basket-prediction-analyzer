# Retail Basket Prediction Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/vu5mcxqya7/badge.svg)](https://ask.thili.ai/b/vu5mcxqya7)

This project provides a Retail Basket Prediction Analyzer for comparing next-item prediction accuracy. It trains both an n-gram model and a small language model on simulated shopping basket data. The tool offers insights into their respective strengths for predicting future purchases.

## What you are building

**Goal.** You will build a tool for retail analysts to compare the next-item prediction accuracy of an n-gram model and a small language model, both trained on simulated shopping basket data.

**Done when.** You will have successfully generated a synthetic dataset, implemented and trained both an n-gram model and a small language model on it, evaluated their next-item prediction performance, and presented a comparison of their results.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Generate a synthetic dataset of shopping baskets with planted itemsets.**  
      Hint: Apply the data generation techniques from 'Teach a Language Model What Sells Together' to create your dataset.
- [ ] **2. Implement the n-gram model to predict next items in a basket.**  
      Hint: Use the n-gram model construction method from 'Build a Next-Word Predictor from Scratch'.
- [ ] **3. Pretrain a small language model (like in the drill) on the basket sequences.**  
      Hint: Train your small language model following the approach in 'Teach a Language Model What Sells Together'.
- [ ] **4. Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).**  
      Hint: Calculate next-item prediction metrics such as accuracy and rank for both models.
- [ ] **5. Present a comparison of the models' performance.**  
      Hint: Summarize and compare the evaluation results to highlight model strengths and weaknesses.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Analyze specific types of itemsets or basket lengths where one model performs better than the other.
- Explore additional next-item prediction metrics beyond accuracy and rank.
- Discuss the practical implications of each model's performance for retail strategies.
