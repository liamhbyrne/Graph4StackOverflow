"""
Multi-layer Perceptron (MLP) model.
"""
import logging

import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, global_mean_pool

from ACL2024.modules.util.custom_logger import setup_custom_logger

log = setup_custom_logger("hetero_gat", logging.INFO)


class MLP(torch.nn.Module):
    """
    Heterogeneous Graph Attentional Network (GAT) model.
    """

    def __init__(
        self, hidden_channels, out_channels, num_layers, dropout, vertex_types, device
    ):
        super().__init__()
        log.info("MODEL: GAT")

        self.convs = torch.nn.ModuleList()

        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.dropout = dropout
        self.device = device

        self.vertex_types = vertex_types

    def forward(self, x_dict, edge_index_dict, batch_dict, post_emb):
        x_dict = {key: x_dict[key] for key in x_dict.keys() if key in self.vertex_types}

        outs = []
        for x, batch in zip(x_dict.values(), batch_dict.values()):
            if len(x):
                outs.append(
                    global_mean_pool(x, batch=batch, size=len(post_emb)).to(self.device)
                )
            else:
                outs.append(torch.zeros(1, x.size(-1)).to(self.device))

        out = torch.cat(outs, dim=1).to(self.device)

        out = torch.cat([out, post_emb], dim=1).to(self.device)

        out = F.dropout(out, p=self.dropout, training=self.training)

        out = self.lin1(out)
        out = F.leaky_relu(out)

        out = self.lin2(out)
        out = F.leaky_relu(out)

        out = self.softmax(out)
        return out
