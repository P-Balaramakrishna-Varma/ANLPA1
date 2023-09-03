## Nueral

| id | hidden_size | lr | epochs | batch |  perplexity | loss | accuracy |
|----|-------------|----|-------|--------|-------------|------|----------|
| 1  | 300 | 0.001 | 10 | 64 | 713 | 6.57 | 0.11 |
| 2  | 300 | 0.001 | 10 | 64 | 733 | 6.59 | 0.06 |
| 3  | 150 | 0.001 | 10 | 64 | 718 | 6.57 | 0.06  |
| 4  | 10 | 0.001 | 10 | 64 | 735 | 6.59 | 0.06  |
| 5 | 300 | 0.00001 | 10 | 64 | 288 | 5.66 | 0.13 |


experiment one idicates overfitting. Increase in validation loss.
dropout layer for regulazation. 0.8
150 0.8 drop
10 0.9 drop
300 perpliexity also achived


## Recurent

| id | seq_len | lr | epochs | batch |  perplexity | loss | accuracy |
|----|---------|----|--------|-------|--------------|------|----------|