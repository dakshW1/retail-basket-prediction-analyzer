# Next-item prediction: n-gram vs small LM (200 held-out cases)

| Model | MRR (95% CI) | Accuracy (hits@1) | hits@3 | hits@10 | Mean rank |
|---|---|---|---|---|---|
| popularity (ignores the basket) | 0.124 (0.095-0.157) | 5.0% | 12.0% | 24.0% | 42.23 |
| 2-gram (k=1.0) | 0.115 (0.087-0.147) | 4.5% | 8.5% | 25.5% | 41.98 |
| co-occurrence counts | 0.183 (0.143-0.228) | 10.5% | 19.5% | 31.5% | 40.53 |
| small LM (distilgpt2, pretrained) | 0.106 (0.079-0.136) | 3.5% | 9.5% | 23.5% | 45.87 |

## MRR by group

| Group | Cases | popularity (ignores the basket) | 2-gram (k=1.0) | co-occurrence counts | small LM (distilgpt2, pretrained) |
|---|---|---|---|---|---|
| basket size 2-3 | 53 | 0.123 | 0.105 | 0.115 | 0.103 |
| basket size 4-6 | 82 | 0.135 | 0.132 | 0.199 | 0.101 |
| basket size 7+ | 65 | 0.112 | 0.101 | 0.217 | 0.114 |
| hidden item is planted | 23 | 0.038 | 0.071 | 0.614 | 0.100 |
| hidden item is ordinary | 177 | 0.135 | 0.120 | 0.127 | 0.106 |

Best overall: **co-occurrence counts**.
