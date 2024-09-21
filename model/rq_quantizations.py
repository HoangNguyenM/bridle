# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# LICENSE file in the root directory of this source tree.

from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class VQEmbedding(nn.Embedding):
    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        ema: bool = False,
        decay: float = 0.99,
        restart_unused_codes: bool = True,
        eps: float = 1e-5,
        training: bool = True,
        init_weight_multiplier: float = 0.0001,
        kmeans_init: bool = False,
    ):
        if training and not kmeans_init:
            # initializing weights in the order of input data variance
            _weight = torch.rand(n_embed + 1, embed_dim) * init_weight_multiplier
            super().__init__(
                n_embed + 1, embed_dim, padding_idx=n_embed, _weight=_weight
            )
        else:
            super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.training = training
        self.kmeans_init = kmeans_init

        # The codebooks are updated with ema and not with backprop
        # This choice is inspired from https://arxiv.org/pdf/2306.08121.pdf for more stable training
        # Details of ema in https://arxiv.org/abs/1711.00937
        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            # padding index is not updated by EMA.
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            # self.weight[-1] is the padding index with zeroes, used for out-of-vocab embedding
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

        if self.kmeans_init:
            self.register_buffer('initted', torch.Tensor([False]))

    @torch.no_grad()
    def compute_distances(self, inputs: torch.Tensor) -> torch.Tensor:
        codebook_t = self.weight[:-1, :].t()
        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(
            *inputs_shape[:-1], -1
        )  # [B, n_embed or n_embed+1]
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        distances = self.compute_distances(inputs)  # [B, n_embed or n_embed+1]
        embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x: torch.Tensor, target_n: int) -> torch.Tensor:
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors: torch.Tensor, idxs: torch.Tensor):
        cluster_size, _ = self._get_cluster_size(vectors, idxs)

        embed_dim = self.weight.shape[-1]
        vectors = vectors.reshape(-1, embed_dim)

        if dist.is_initialized():
            # mask reduced from all ranks to rank 0
            mask = cluster_size == 0
            dist.reduce(mask, 0, op=dist.ReduceOp.MAX)
            random_ids = torch.randint(
                0,
                vectors.size(0),
                (self.weight.size(0) - 1,),
                device=vectors.device,
            )
            random_tensors = torch.index_select(  # (codebook_cardinality, emb_dim)
                vectors, 0, random_ids
            )
            weight_update = torch.where(
                mask[:, None], random_tensors, self.weight[:-1][:]
            )
            dist.broadcast(weight_update, src=0)
            self.weight[:-1, :] = weight_update
            self.restart_unused_codes = False

    @torch.no_grad()
    def _update_embedding(
        self,
        vectors: torch.Tensor,
        idxs: torch.Tensor,
    ):
        cluster_size, one_hot_idxs = self._get_cluster_size(vectors, idxs)
        if dist.is_initialized():
            dist.all_reduce(cluster_size)

        self._param_ema_update(self.cluster_size_ema, cluster_size, self.decay)
        embed_sum = one_hot_idxs @ vectors
        if dist.is_initialized():
            dist.all_reduce(embed_sum)

        self._param_ema_update(self.embed_ema, embed_sum, self.decay)

        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )

        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, input: torch.Tensor):
        if self.kmeans_init:
            self.init_embed_(input.reshape(-1, self.embed_dim))
            self.kmeans_init = False

        embed_idxs = self.find_nearest_embedding(input)
        # perform ema update
        if self.ema and self.training:
            self._update_embedding(input, embed_idxs)

        # perform restart unused codes
        if self.restart_unused_codes:
            self._update_buffers(input, embed_idxs)

        embeds = self.embed(embed_idxs)

        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds

    @torch.no_grad()
    def _get_cluster_size(
        self, vectors: torch.Tensor, idxs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        n_vectors = vectors.shape[0]

        one_hot_idxs = vectors.new_zeros(n_embed, n_vectors)
        one_hot_idxs.scatter_(
            dim=0, index=idxs.unsqueeze(0), src=vectors.new_ones(1, n_vectors)
        )
        cluster_size = one_hot_idxs.sum(dim=1)
        return cluster_size, one_hot_idxs

    @torch.no_grad()
    def _param_ema_update(
        self, params: torch.Tensor, new_value: torch.Tensor, decay: float
    ):
        params.data.mul_(decay).add_(new_value, alpha=(1 - decay))

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        print("Performing Kemans init for codebook")
        embed, cluster_size = kmeans(data, self.n_embed, 10, use_cosine_sim=True)
        if dist.is_initialized():
            dist.all_reduce(cluster_size)
            dist.all_reduce(embed, op=dist.ReduceOp.AVG)
            
        self.weight[:-1, :].data.copy_(embed)
        self.cluster_size_ema.data.copy_(cluster_size)
        self.kmeans_init = False


class RQBottleneck(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.

    Arguments:
        latent_shape (Tuple[int]): the shape of latents, denoted (D)
        code_shape (Tuple[int]): the shape of codes, denoted (d)
        n_embed (int, List, or Tuple): the number of embeddings (i.e., the size of codebook)
            If isinstance(n_embed, int), the sizes of all codebooks are same.
        init_weight_multiplier: float: the initial weight multiplier for initialization.
        training (bool): if True, used to init with weight multiplier
        shared_codebook (bool): If True, codebooks are shared in all location. If False,
            uses separate codebooks along the ``depth'' dimension. (default: False)
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(
        self,
        latent_shape: List[int],  # latent / bottleneck embedding dimension, eg. [32]
        code_shape: List[int],  # depth of codebooks, eg. [3]
        n_embed: List[int],  # number of embeddings in each codebook, eg. [256,256,256]
        kmeans_init: bool = False,
        init_weight_multiplier: float = 0.0001,
        commitment_loss_weight: float = 0.25,
        training: bool = True,
        ema: bool = False,
        decay: float = 0.99,
        shared_codebook: bool = False,
        restart_unused_codes: bool = True,
    ):
        super().__init__()

        if not len(code_shape) == len(latent_shape) == 1:
            raise ValueError("incompatible code shape or latent shape")

        # residual quantization does not divide feature dims for quantization.
        embed_dim = latent_shape[-1]

        self.latent_shape = torch.Size(latent_shape)
        self.code_shape = torch.Size(code_shape)

        self.shared_codebook = shared_codebook
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError(
                    "Shared codebooks are incompatible \
                                    with list types of momentums or sizes: Change it into int"
                )

        self.restart_unused_codes = restart_unused_codes
        self.commitment_loss_weight = commitment_loss_weight

        self.n_embed = (
            n_embed
            if isinstance(n_embed, Iterable)
            else [n_embed for _ in range(self.code_shape[-1])]
        )

        self.decay = (
            decay
            if isinstance(decay, Iterable)
            else [decay for _ in range(self.code_shape[-1])]
        )
        assert len(self.n_embed) == self.code_shape[-1]
        assert len(self.decay) == self.code_shape[-1]

        if self.shared_codebook:
            codebook0 = VQEmbedding(
                self.n_embed[0],
                embed_dim,
                ema=ema,
                decay=self.decay[0],
                restart_unused_codes=restart_unused_codes,
                init_weight_multiplier=init_weight_multiplier,
                kmeans_init=kmeans_init,
                training=training,
            )

            self.codebooks = nn.ModuleList(
                [codebook0 for _ in range(self.code_shape[-1])]
            )
        else:
            codebooks = [
                VQEmbedding(
                    self.n_embed[idx],
                    embed_dim,
                    ema=ema,
                    decay=self.decay[idx],
                    restart_unused_codes=restart_unused_codes,
                    init_weight_multiplier=init_weight_multiplier,
                    kmeans_init=kmeans_init,
                    training=training,
                )
                for idx in range(self.code_shape[-1])
            ]
            self.codebooks = nn.ModuleList(codebooks)

    def quantize(self, x: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        torch.Tensor,
        List[torch.Tensor],
    ]:
        r"""
        Return list of quantized features and the selected codewords by the residual quantization.
        The code is selected by the residuals between x and quantized features by the previous codebooks.

        Arguments:
            x (Tensor): bottleneck feature maps to quantize.

        Returns:
            quant_list (list): list of sequentially aggregated and quantized feature maps by codebooks.
            codes (LongTensor): codewords index, corresponding to quants.
            loss_list (list): list of residual quantization ( rqvae loss per codebook ) losses.

        Shape:
            - x: (B, embed_dim) for 1D embedding inputs OR (B, token, embed_dim) for 2D image inputs
            - quant_list[i]: (B, embed_dim) for 1D embedding inputs or (B, token, embed_dim) for 2D image inputs
            - codes: (B, d) for 1D embedding inputs or (B, token, d) for 2D image inputs
        """

        residual_feature = x

        quant_list: List[torch.Tensor] = []
        code_list: List[torch.Tensor] = []
        loss_list: List[torch.Tensor] = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.code_shape[-1]):
            quant, code = self.codebooks[i](residual_feature)
            embedding_loss = F.mse_loss(
                residual_feature.detach(), quant, reduction="mean"
            )
            encoding_loss = F.mse_loss(
                residual_feature, quant.detach(), reduction="mean"
            )
            partial_loss = embedding_loss + self.commitment_loss_weight * encoding_loss
            loss_list.append(partial_loss)
            residual_feature = residual_feature - quant
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants)
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes, loss_list

    def forward(self, z: torch.Tensor):
        bsz, seq_len, hidden_dim = z.shape
        z_flattened = z.reshape(-1, hidden_dim)

        quant_list, codes, loss_list = self.quantize(z_flattened)

        loss = torch.mean(torch.stack(loss_list))
        z_q = z_flattened + (quant_list[-1] - z_flattened).detach()
        z_q = z_q.view(bsz, seq_len, hidden_dim)
        codes = codes.view(bsz, seq_len, -1)
        return z_q, loss, codes

    @torch.no_grad()
    def eval_codebooks(self, z: torch.Tensor):
        bsz, seq_len, hidden_dim = z.shape
        z_flattened = z.reshape(-1, hidden_dim)

        residual_feature = z_flattened

        quant_list: List[torch.Tensor] = []
        code_list: List[torch.Tensor] = []

        restore_training = False

        # turn off EMA update if training is on
        if self.codebooks[0].training:
            restore_training = True
            for i in range(self.code_shape[-1]):
                self.codebooks[i].training = False

        for i in range(self.code_shape[-1]):
            quant, code = self.codebooks[i](residual_feature)

            residual_feature = residual_feature - quant

            quant_list.append(quant)
            code_list.append(code.unsqueeze(-1))

        # restore EMA update if training is on
        if restore_training:
            for i in range(self.code_shape[-1]):
                self.codebooks[i].training = True

        codes = torch.cat(code_list, dim=-1)
        codes = codes.view(bsz, seq_len, -1)
        
        quants = torch.stack(quant_list, dim=-1)
        quants = quants.view(bsz, seq_len, -1, self.code_shape[-1])
        return quants, codes
    
def l2norm(t):
    return F.normalize(t, p=2, dim=-1)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]

def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim = samples.shape[-1]

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            # samples (n, d) -> (n, 1, d) and means (c, d) -> (1, c, d)
            diffs = samples.unsqueeze(1) - means.unsqueeze(0)
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=samples.dtype)
        new_means.scatter_add_(0, buckets.unsqueeze(1).repeat(1, dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins