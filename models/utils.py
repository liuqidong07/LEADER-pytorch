# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrastive_Loss(nn.Module):

    def __init__(self, tau=1, project=False, in_dim_1=None, in_dim_2=None, out_dim=None) -> None:
        super().__init__()
        self.tau = tau
        self.project = project

        if project:
            if not in_dim_1:
                return ValueError
            self.x_projector = nn.Linear(in_dim_1, out_dim)
            self.y_projector = nn.Linear(in_dim_2, out_dim)


    def forward(self, X, Y):
        
        if self.project:
            X = self.x_projector(X)
            Y = self.y_projector(Y)

        loss = self.compute_cl(X, Y) + self.compute_cl(Y, X)

        return loss
    

    def compute_cl(self, X, Y):

        '''
        X: (bs, hidden_size), Y: (bs, hidden_size)
        tau: the temperature factor
        '''
        #sim_matrix = X.mm(Y.t())    # (bs, bs)
        sim_matrix = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=2)
        pos = torch.exp(torch.diag(sim_matrix) / self.tau).unsqueeze(0)   # (1, bs)
        neg = torch.sum(torch.exp(sim_matrix / self.tau), dim=0) - pos     # (1, bs)
        loss = - torch.log(pos / neg)
        loss = loss.view(X.shape[0], -1)

        return loss