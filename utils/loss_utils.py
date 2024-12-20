import torch
import torch.nn as nn
import numpy as np

class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()
    def forward(self, risk_scores, events, survival_times):
        idx = survival_times.sort(descending=True)[1]
        events = events[idx]
        risk_scores = risk_scores[idx]
        events = events.float()
        events = events.view(-1)
        risk_scores = risk_scores.view(-1)
        uncensored_likelihood = risk_scores - risk_scores.exp().cumsum(0).log()
        censored_likelihood = uncensored_likelihood * events
        num_observed_events = events.sum()
        neg_likelihood = -censored_likelihood.sum()/num_observed_events
        return neg_likelihood
    
class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)
        

class CoxSurvLoss(object):
    def __call__(self, hazards, time, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(time)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = time[j] >= time[i]

        c = torch.FloatTensor(c).to(hazards.device)
        R_mat = torch.FloatTensor(R_mat).to(hazards.device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        # print(loss_cox)
        # print(R_mat)
        return loss_cox
    

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    
    #!! here uncensored_loss means event happens(death/progression)
    # uncensored_loss = -c * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    # censored_loss = - (1 - c) * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    # neg_l = censored_loss + uncensored_loss
    # loss = (1-alpha) * neg_l + alpha * uncensored_loss
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss