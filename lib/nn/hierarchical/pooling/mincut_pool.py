import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import tsl
from tgp.connect import DenseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import Select, SelectOutput
from tgp.src import DenseSRCPooling, PoolingOutput
from tgp.utils.losses import mincut_loss, orthogonality_loss
from tgp.utils.ops import postprocess_adj_pool_dense
from tsl.nn.layers.base import NodeEmbedding


class NodeEmbeddingSelect(Select):
    """Dense select operator using a prior learnable node embedding matrix
    with Gumbel-Softmax sampling during training.

    Unlike :class:`~tgp.select.MLPSelect`, the assignment logits are
    position-dependent (one learnable vector per node) rather than
    feature-dependent. At training time, Gumbel-Softmax produces
    (optionally hard) one-hot-like samples; at test time, either argmax
    (hard) or plain softmax is used.
    """

    is_dense: bool = True

    def __init__(
        self,
        n_nodes: int,
        k: int,
        hard: bool = True,
        temp: float = 1.0,
        temp_decay: float = 0.99995,
        temp_min: float = 0.05,
        s_inv_op: str = "transpose",
    ):
        super().__init__()
        self.k = k
        self.hard = hard
        self._temp = temp
        self.temp_decay = temp_decay
        self.temp_min = temp_min
        self.s_inv_op = s_inv_op

        self.assignment_logits = NodeEmbedding(n_nodes=n_nodes, emb_size=k)

    def reset_parameters(self):
        self.assignment_logits.reset_emb()

    @property
    def temp(self) -> float:
        return self._temp

    def anneal_temperature(self):
        """Decay temperature by one step (call once per forward during training)."""
        self._temp = max(self._temp * self.temp_decay, self.temp_min)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        **kwargs,
    ) -> SelectOutput:
        logits = self.assignment_logits()  # [N, K]

        # Expand for batched input [B, N, F] -> logits [B, N, K]
        if x.dim() == 3:
            logits = logits.unsqueeze(0).expand(x.size(0), -1, -1)

        if self.training:
            self.anneal_temperature()
            s = F.gumbel_softmax(logits / self._temp, hard=self.hard, dim=-1)
        else:
            if self.hard:
                s = F.one_hot(
                    torch.argmax(logits, dim=-1),
                    num_classes=self.k,
                ).float()
            else:
                s = F.softmax(logits / self._temp, dim=-1)

        # Prevent exact zeros in the assignment matrix. Downstream,
        # compute_aggregation_matrix cascades these matrices via matmul;
        # exact zeros can produce all-zero rows in C, making the
        # reconciliation matrix Q singular (lstsq / SVD failure).
        s = s + tsl.epsilon

        if mask is not None:
            s = s * mask.unsqueeze(-1)

        return SelectOutput(s=s, s_inv_op=self.s_inv_op, in_mask=mask)


class GumbelMinCutPooling(DenseSRCPooling):
    """MinCut pooling with prior learnable node embeddings and Gumbel-Softmax.

    This reproduces the original assignment mechanism (position-dependent
    logits + Gumbel-Softmax + temperature annealing) within the TGP
    Select-Reduce-Connect framework, keeping the MinCut and orthogonality
    regularisation losses.
    """

    def __init__(
        self,
        n_nodes: int,
        k: int,
        hard: bool = True,
        temp: float = 1.0,
        temp_decay: float = 0.99995,
        temp_min: float = 0.05,
        cut_loss_coeff: float = 1.0,
        ortho_loss_coeff: float = 1.0,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        adj_transpose: bool = False,
    ):
        selector = NodeEmbeddingSelect(
            n_nodes=n_nodes,
            k=k,
            hard=hard,
            temp=temp,
            temp_decay=temp_decay,
            temp_min=temp_min,
        )
        super().__init__(
            selector=selector,
            reducer=BaseReduce(),
            lifter=BaseLift(matrix_op="precomputed"),
            connector=DenseConnect(
                remove_self_loops=remove_self_loops,
                degree_norm=degree_norm,
                adj_transpose=adj_transpose,
            ),
            adj_transpose=adj_transpose,
            batched=True,
            sparse_output=False,
        )

        self.cut_loss_coeff = cut_loss_coeff
        self.ortho_loss_coeff = ortho_loss_coeff

    def compute_loss(self, adj, s, adj_pool) -> dict:
        """MinCut + orthogonality losses computed on the *soft* assignment."""
        # For loss computation, use a softmax of the raw logits (no Gumbel noise)
        # to get a smooth gradient signal, same as the original implementation.
        raw_logits = self.selector.assignment_logits()  # [N, K]
        if s.dim() == 3:
            raw_logits = raw_logits.unsqueeze(0).expand(s.size(0), -1, -1)
        s_soft = F.softmax(raw_logits / self.selector.temp, dim=-1)

        adj_soft = self.connector.dense_connect(adj=adj, s=s_soft)
        cut = mincut_loss(adj, s_soft, adj_soft) * self.cut_loss_coeff
        ortho = orthogonality_loss(s_soft) * self.ortho_loss_coeff
        return {"cut_loss": cut, "ortho_loss": ortho}

    def forward(
        self,
        x: Tensor,
        adj: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        so: Optional[SelectOutput] = None,
        mask: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        batch_pooled: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> PoolingOutput:
        if lifting:
            batch_orig = batch if batch is not None else so.batch
            return self.lift(
                x_pool=x, so=so, batch=batch_orig, batch_pooled=batch_pooled
            )

        # Ensure dense batched inputs
        x, adj, mask = self._ensure_batched_inputs(
            x=x,
            edge_index=adj,
            edge_weight=edge_weight,
            batch=batch,
            mask=mask,
        )

        # Select (Gumbel-Softmax on node embedding logits)
        so = self.select(x=x, mask=mask)

        # Reduce: S^T @ X
        x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

        # Connect: S^T @ A @ S
        adj_pool = self.connector.dense_connect(adj=adj, s=so.s)

        # Loss (uses soft assignment, not the Gumbel sample)
        loss = self.compute_loss(adj, so.s, adj_pool) if self.training else {}

        # Post-process adjacency (remove self-loops, degree normalize)
        adj_pool = postprocess_adj_pool_dense(
            adj_pool,
            remove_self_loops=self.connector.remove_self_loops,
            degree_norm=self.connector.degree_norm,
            adj_transpose=self.connector.adj_transpose,
        )

        return PoolingOutput(
            x=x_pooled,
            edge_index=adj_pool,
            batch=batch_pooled,
            so=so,
            loss=loss,
        )
