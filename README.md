# pytorch-transformer-reranker-kldiv
pytorch implementation of reranker using kullback-leibler divergence loss.

### Information about the code:

1. This is a PyTorch - PyTorch Lightning based reranker using the Kullback-Leibler Divergence Loss (I love this explanation so i will share it here: https://qr.ae/prWypg)
2. Your `BATCH_SIZE` should be equal to the number of candidates, so a single batch should contains all candidates.
3. The dataloader should yield as follows: 
  - `batch[0]` -> Tensor of size (`B x (len(I)+len(C)+len(padding)`). concatenated tensor of the Input I and Candidate C plus the padding. 
  - `batch[1]` -> Tensor of size (`B x (len(I)+len(C)+len(padding))`). contains "segments", marking the boundary between the Input I, Candidate C, and padding. 
  - `batch[2]` -> Tensor of size (`B`). contains the BLEU or any other metrics for the reranking.
4. pytorch_lightning `predict()` function automatically pick the best prediction.
