

"""
GraphSAGE model for heterogeneous graphs.
"""
import logging

import torch
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

from ACL2024.modules.util.custom_logger import setup_custom_logger

log = setup_custom_logger("hetero_GraphSAGE", logging.INFO)


class HeteroGraphSAGE(torch.nn.Module):
    """
    Heterogeneous GraphSAGE model.
    """

    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        log.info("MODEL: GraphSAGE")
        self.convs = torch.nn.ModuleList()

        # Create Graph Attentional layers
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("tag", "describes", "question"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("tag", "describes", "answer"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("tag", "describes", "comment"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("module", "imported_in", "question"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("module", "imported_in", "answer"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("question", "rev_describes", "tag"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("answer", "rev_describes", "tag"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("comment", "rev_describes", "tag"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("question", "rev_imported_in", "module"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                    ("answer", "rev_imported_in", "module"): SAGEConv(
                        (-1, -1), hidden_channels, add_self_loops=False
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x_dict, edge_index_dict, batch_dict, post_emb):
        for conv in self.convs:
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

