## Nueral

| id | hidden_size | lr | epochs | batch |  perplexity | loss | accuracy |
|----|----|--------|-------|--------------|------|----------|
| 1  | 300 | 0.01 | 10 | 64 | 13909 | 9.54 | 0.05 |
| 2  | 300 | 0.001 | 10 | 64 | 13509 | 9.51 | 0.089 |
| 3  | 300 | 0.001 | 20 | 64 | 13358 | 9.499 | 0.1 |
| 4  | 300 | 0.01 | 10 | 64 | 13827 | 9.53 | 0.065 | 0.5 dropout
| 5 | 500 | 0.001 | 10 | 64 | 13705 | 9.52 | 0.07 |

experiment one idicates overfitting.
lowering learning rate || using dropout
experiment 2 lower learning rate still drop try increasing epochs 20 (3)
experiment 3 stabilized. no learning

## Recurent

| id | seq_len | lr | epochs | batch |  perplexity | loss | accuracy |
|----|---------|----|--------|-------|--------------|------|----------|