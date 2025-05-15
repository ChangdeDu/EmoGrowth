import torch
import torch.nn as nn


class LinkPredictionLoss_cosine(nn.Module):
    def __init__(self):
        super(LinkPredictionLoss_cosine, self).__init__()

    def forward(self, emb, adj):
        '''
        Parameters
        ----------
        emb : Tensor
            An MxE tensor, the embedding of the ith node is stored in emb[i,:].
        adj : Tensor
            An MxM tensor, adjacent matrix of the graph.

        Returns
        -------
        loss : float
            The link prediction loss.
        '''
        emb_norm = emb.norm(dim=1, keepdim=True)
        emb_norm = emb / (emb_norm + 1e-6)
        adj_pred = torch.matmul(emb_norm, emb_norm.t())
        loss = torch.mean(torch.pow(adj - adj_pred, 2))

        return loss