from typing import Optional, Sequence
import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

def log_cosh(x):
    return x + F.softplus(-2. * x) - torch.log(torch.tensor(2.0))
    
class CrossEntropyLogCosh(torch.nn.L1Loss):
    __constants__ = ['reduction','loss_lambda','loss_gamma','quantiles','loss_kappa','domain_weight','domain_dim']

    def __init__(self, 
                 reduction: str = 'mean', 
                 loss_lambda: float = 0.2, 
                 loss_gamma: float = 0.2, 
                 quantiles: list = [],
             ) -> None:
        
        super(CrossEntropyLogCosh, self).__init__(None, None, reduction)
        self.loss_lambda = loss_lambda
        self.loss_gamma = loss_gamma
        self.quantiles = quantiles
        
    def forward(self, 
                input_cat: Tensor, y_cat: Tensor, 
                input_reg: Tensor, y_reg: Tensor, 
                input_reg_wnu: Tensor, y_reg_wnu: Tensor, 
                device) -> Tensor:

        ## classification term
        loss_cat  = torch.tensor([0.0]).to(device)
        if input_cat.nelement():
            loss_cat = torch.nn.functional.cross_entropy(input_cat, y_cat, reduction = self.reduction)

        x_reg         = input_reg - y_reg
        x_reg_wnu     = input_reg_wnu - y_reg_wnu
        loss_mean     = 0
        loss_mean2    = 0
        loss_quant    = 0
        loss_reg      = torch.tensor([0.0]).to(device)

        #Log-cosh loss only for the genjet_pt_wnu
        if input_reg_wnu.nelement():
            loss_mean2 += log_cosh(x_reg_wnu)
        
        #Log-cosh + quantile losses for the genjet_pt
        if input_reg.nelement():
            
            ## compute loss
            for idx, q in enumerate(self.quantiles):
                x_reg_eval = x_reg[:, idx] if len(self.quantiles) > 1 else x_reg
                if q <= 0:
                    loss_mean += log_cosh(x_reg_eval)
                elif q > 0:
                    loss_quant += (q * x_reg_eval * (x_reg_eval >= 0)) + ((q - 1) * x_reg_eval * (x_reg_eval < 0))
                            
        ## reduction
        if self.reduction == 'mean':
            loss_quant = loss_quant.mean()
            loss_mean  = loss_mean.mean()
            loss_mean2 = loss_mean2.mean()
        elif self.reduction == 'sum':
            loss_quant = loss_quant.sum()
            loss_mean  = loss_mean.sum()
            loss_mean2 = loss_mean2.sum()
            
        ## composition
        loss_reg = self.loss_lambda * (loss_mean + loss_mean2) + self.loss_gamma * loss_quant
            
        return loss_cat + loss_reg, loss_cat, loss_reg
