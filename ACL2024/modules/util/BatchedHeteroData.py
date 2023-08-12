from typing import Optional, Any

from torch_geometric.data.hetero_data import NodeOrEdgeStorage
from torch_geometric.data import HeteroData


class BatchedHeteroData(HeteroData):
    """ """

    def __cat_dim__(
        self,
        key: str,
        value: Any,
        store: Optional[NodeOrEdgeStorage] = None,
        *args,
        **kwargs
    ):
        if key in ["question_emb", "answer_emb", "label"]:
            return None
        return super().__cat_dim__(key, value)
