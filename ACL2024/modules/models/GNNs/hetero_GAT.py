"""
Heterogeneous Graph Attentional Network (GAT) model.
"""
import logging

import torch
from torch_geometric.nn import HeteroConv

from ACL2024.modules.util.custom_logger import setup_custom_logger

log = setup_custom_logger("hetero_gat", logging.INFO)


class HeteroGAT(torch.nn.Module):
    """
    Heterogeneous Graph Attentional Network (GAT) model.
    """

    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        log.info("MODEL: GAT")

        self.convs = torch.nn.ModuleList()

        # Create Graph Attentional layers
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("tag", "describes", "question"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False, heads=6
                    ),
                    ("tag", "describes", "answer"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False, heads=6
                    ),
                    ("tag", "describes", "comment"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False, heads=6
                    ),
                    ("module", "imported_in", "question"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False, heads=6
                    ),
                    ("module", "imported_in", "answer"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False, heads=6
                    ),
                    ("question", "rev_describes", "tag"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False, heads=6
                    ),
                    ("answer", "rev_describes", "tag"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False, heads=6
                    ),
                    ("comment", "rev_describes", "tag"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False, heads=6
                    ),
                    ("question", "rev_imported_in", "module"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False, heads=6
                    ),
                    ("answer", "rev_imported_in", "module"): GATConv(
                        (-1, -1), hidden_channels, add_self_loops=False, heads=6
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x_dict, edge_index_dict, batch_dict, post_emb):
        x_dict = {
            key: x_dict[key]
            for key in x_dict.keys()
            if key in ["question", "answer", "comment", "tag", "module"]
        }

        for conv in self.convs:
            break
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            x_dict = {
                key: F.dropout(x, p=DROPOUT, training=self.training)
                for key, x in x_dict.items()
            }

        outs = []

        for x, batch in zip(x_dict.values(), batch_dict.values()):
            if len(x):
                outs.append(
                    global_mean_pool(x, batch=batch, size=len(post_emb)).to(device)
                )
            else:
                outs.append(torch.zeros(1, x.size(-1)).to(device))

        out = torch.cat(outs, dim=1).to(device)

        out = torch.cat([out, post_emb], dim=1).to(device)

        out = F.dropout(out, p=DROPOUT, training=self.training)

        out = self.lin1(out)
        out = F.leaky_relu(out)

        out = self.lin2(out)
        out = F.leaky_relu(out)

        out = self.softmax(out)
        return out