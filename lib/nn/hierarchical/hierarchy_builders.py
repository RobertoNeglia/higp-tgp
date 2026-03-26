import torch
from torch import nn
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor

from lib.nn.hierarchical.pooling import GumbelMinCutPooling, MinCutPooling

from tsl.utils import ensure_list

POOLERS = {
    "gumbel": GumbelMinCutPooling,
    "mlp": MinCutPooling,
}


class MinCutHierarchyBuilder(nn.Module):
    r"""Hierarchy encoder with configurable MinCut pooling variant.

    Args:
        pooler: ``'gumbel'`` for NodeEmbedding + Gumbel-Softmax + temperature
            annealing, ``'mlp'`` for TGP's feature-dependent MLPSelect.
    """

    def __init__(
        self,
        n_nodes: int,
        hidden_size: int,
        n_clusters: float,
        n_levels: int = 1,
        pooler: str = "gumbel",
        temp_decay: float = 0.99995,
        hard: bool = True,
    ):
        super(MinCutHierarchyBuilder, self).__init__()

        self.n_levels = n_levels
        self.pooler_name = pooler
        pooler_cls = POOLERS[pooler]
        input_nodes = n_nodes
        pooling_layers = []
        n_clusters = ensure_list(n_clusters)
        if len(n_clusters) != n_levels - 2:
            assert len(n_clusters) == 1
            n_clusters = n_clusters * (n_levels - 2)
        for i in range(n_levels - 2):
            if pooler == "gumbel":
                pooling_layers.append(
                    pooler_cls(
                        n_nodes=input_nodes,
                        k=n_clusters[i],
                        hard=hard,
                        temp_decay=temp_decay,
                        remove_self_loops=True,
                        degree_norm=True,
                        adj_transpose=False,
                    )
                )
            else:  # mlp
                pooling_layers.append(
                    pooler_cls(
                        in_channels=hidden_size,
                        k=n_clusters[i],
                        remove_self_loops=True,
                        degree_norm=True,
                        adj_transpose=False,
                    )
                )
            input_nodes = n_clusters[i]

        self.pooling_layers = nn.ModuleList(pooling_layers)

    def forward(self, emb, edge_index, edge_weight=None):
        # emb: [nodes features] or [batch nodes features]
        if isinstance(edge_index, SparseTensor):
            adj = edge_index.to_dense()
        else:
            adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0].T

        # force the graph to be undirected
        adj = torch.max(adj, adj.T)

        # expand to batched dense [B, N, N] when emb is batched
        if emb.dim() == 3 and adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(emb.size(0), -1, -1)

        embs = [emb]
        adjs = [adj]
        selects = [None]
        sizes = [emb.size(-2)]
        min_cut_loss = 0.0
        reg_loss = 0.0
        for i in range(self.n_levels - 2):
            pool_out = self.pooling_layers[i](x=embs[i], adj=adjs[i])
            selects.append(pool_out.so.s)
            embs.append(pool_out.x)
            adjs.append(pool_out.edge_index)
            sizes.append(pool_out.x.size(-2))
            if pool_out.has_loss:
                min_cut_loss += pool_out.loss["cut_loss"]
                reg_loss += pool_out.loss["ortho_loss"]

        # add the last level (global node via mean aggregation)
        embs.append(emb.mean(-2, keepdim=True))
        adjs.append(None)
        if self.n_levels > 2 and emb.dim() == 3:
            s_tot = torch.ones(emb.size(0), sizes[-1], 1, device=emb.device)
        else:
            s_tot = torch.ones(sizes[-1], 1, device=emb.device)
        selects.append(s_tot)
        sizes.append(1)
        return embs, adjs, selects, sizes, (min_cut_loss, reg_loss)
