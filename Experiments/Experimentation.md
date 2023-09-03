## Nueral

| id | hidden_size | lr | epochs | batch |  perplexity | loss | accuracy |
|----|-------------|----|-------|--------|-------------|------|----------|
| 1  | 300 | 0.001 | 10 | 64 | 713 | 6.57 | 0.11 |
| 2  | 300 | 0.001 | 10 | 64 | 733 | 6.59 | 0.06 |
| 3  | 150 | 0.001 | 10 | 64 | 718 | 6.57 | 0.06  |
| 4  | 10 | 0.001 | 10 | 64 | 735 | 6.59 | 0.06  |
| 5 | 300 | 0.00001 | 10 | 64 | 288 | 5.66 | 0.13 |


1) experiment one idicates overfitting. Increase in validation loss.  
2) dropout layer for regulazation. 0.8. Still Overfitting.
3) 150 0.8 drop. Still overfitting.
4) 10 0.9 drop. Still overfitting.
5) Reducing learning instead of adding regularization.


## Recurent

| id | seq_len | lr | epochs | batch |  perplexity | loss | accuracy |
|----|---------|----|--------|-------|--------------|------|----------|
| 1  | 15 | 0.001 | 5 | 1024 | 110 | 4.708 | 0.16 |



## Decoder based 

| id | seq_len | lr | epochs | batch |  perplexity | loss | accuracy | heads |
|----|---------|----|--------|-------|--------------|------|----------|-------|
| 1  | 15 | 0.001 | 5 | 1024 | 126 | 4.840 | 0.13 | 1 |
| 2 | 15 | 0.001 | 5 | 1024 | 84 | 4.43 | 0.17 | 3 |


## Conclustion
Decoder better than reccurent better than nueral language model.