import dgl.nn.pytorch.conv as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn import GraphConv

class HyperparameterGiver:
    def __init__(self):
        # self.patience = 200 unused due to disabling EarlyStopping
        self.learning_rate = None
        self.dropout = None
        self.hidden_size = None
        self.l2_regularization = None
class gcnHG(HyperparameterGiver):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.01
        self.dropout = 0.8
        self.hidden_size = 64
        self.l2_regularization = 0.001

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics = None

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.metrics.learning_rate)

class GCN_Leo(CustomModel):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.metrics = gcnHG()
        self.layers = nn.ModuleList()
        # two-layer GCN
        hid_size = self.metrics.hidden_size
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(self.metrics.dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
    
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_size):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, out_size, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h