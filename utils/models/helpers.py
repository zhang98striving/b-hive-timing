import numpy as np
import torch
import torch.nn as nn


class InputConv(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1, norm=True, **kwargs):
        super(InputConv, self).__init__(**kwargs)

        self.norm = norm
        self.lin = torch.nn.Conv1d(in_chn, out_chn, kernel_size=1)
        self.bn = torch.nn.BatchNorm1d(out_chn, eps=0.001, momentum=0.6)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.norm:
            x = self.dropout(self.bn(self.act(self.lin(x))))
        else:
            x = self.act(self.lin(x))
        return x


class LinLayer(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1, **kwargs):
        super(LinLayer, self).__init__(**kwargs)

        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.bn = torch.nn.BatchNorm1d(out_chn, eps=0.001, momentum=0.6)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.bn(self.act(self.lin(x))))
        return x


class InputProcess(nn.Module):
    def __init__(self, 
                 input_dims,
                 cpf_conv,
                 npf_conv,
                 vtx_conv, 
                 **kwargs
        ):
        
        super(InputProcess, self).__init__(**kwargs)

        cpf_conv[:0] = [input_dims[0][1]]
        self.cpf_bn = torch.nn.BatchNorm1d(cpf_conv[0], eps=0.001, momentum=0.6)
        self.cpf_conv = nn.ModuleList([InputConv(cpf_conv[i], cpf_conv[i+1]) for i in range(len(cpf_conv) - 2)])
        self.cpf_conv.append(InputConv(cpf_conv[-2], cpf_conv[-1], norm=False))

        npf_conv[:0] = [input_dims[1][1]]
        self.npf_bn = torch.nn.BatchNorm1d(npf_conv[0], eps=0.001, momentum=0.6)
        self.npf_conv = nn.ModuleList([InputConv(npf_conv[i], npf_conv[i+1]) for i in range(len(npf_conv) - 2)])
        self.npf_conv.append(InputConv(npf_conv[-2], npf_conv[-1], norm=False))

        vtx_conv[:0] = [input_dims[2][1]]
        self.vtx_bn = torch.nn.BatchNorm1d(vtx_conv[0], eps=0.001, momentum=0.6)
        self.vtx_conv = nn.ModuleList([InputConv(vtx_conv[i], vtx_conv[i+1]) for i in range(len(vtx_conv) - 2)])
        self.vtx_conv.append(InputConv(vtx_conv[-2], vtx_conv[-1], norm=False))

    def forward(self, cpf, npf, vtx):
        cpf = self.cpf_bn(torch.transpose(cpf, 1, 2))
        for conv in self.cpf_conv:
            cpf = conv(cpf)
        cpf = torch.transpose(cpf, 1, 2)

        npf = self.npf_bn(torch.transpose(npf, 1, 2))
        for conv in self.npf_conv:
            npf = conv(npf)
        npf = torch.transpose(npf, 1, 2)

        vtx = self.vtx_bn(torch.transpose(vtx, 1, 2))
        for conv in self.vtx_conv:
            vtx = conv(vtx)
        vtx = torch.transpose(vtx, 1, 2)

        return cpf, npf, vtx


class DenseClassifier(nn.Module):
    def __init__(self, dense_clas_dim, **kwargs):
        super(DenseClassifier, self).__init__(**kwargs)

        self.LinLayers = nn.ModuleList([LinLayer(dense_clas_dim[i], dense_clas_dim[i+1]) for i in range(len(dense_clas_dim) - 1)])

    def forward(self, x):

        for layer in self.LinLayers:
            x = layer(x)
        
        return x
