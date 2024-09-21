# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# LICENSE file in the root directory of this source tree.

from typing import Iterable
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.distributed as dist

import util.misc as misc


def eval_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    device: torch.device, epoch: int,
                    log_writer=None,
                    args=None):
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Codebook Eval'
    print_freq = 500

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    flattened_norms = [[] for _ in range(args.codebook_set)]
    code_collection = [torch.zeros(args.code_num, dtype=torch.long, device=device) for _ in range(args.codebook_set)]

    for data_iter_step, (samples, _labels, _vids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            z_q, enc_indices = model(samples, eval_codebooks=True)

        if len(z_q.shape) == 3:
            z_q = z_q.unsqueeze(-1)
            enc_indices = enc_indices.unsqueeze(-1)

        for i in range(args.codebook_set):
            codes = enc_indices[:, :, i].flatten()
            code_collection[i] += torch.bincount(codes, minlength=args.code_num)

            norms = torch.norm(z_q[..., i], dim=-1).flatten()
            flattened_norms[i].append(norms)

        torch.cuda.synchronize()

    norms_collection = []
    for i in range(args.codebook_set):
        all_norms = torch.cat(flattened_norms[i])
        all_norms_global = [torch.zeros_like(all_norms) for _ in range(dist.get_world_size())]
        dist.all_gather(all_norms_global, all_norms)
        all_norms_global = torch.cat(all_norms_global)
        norms_collection.append(all_norms_global)
    
    for i in range(args.codebook_set):
        dist.all_reduce(code_collection[i], op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        make_norm_fig(args, norms_collection, epoch)

    stats = calc_code_stats(code_collection)
    for stat, value in stats.items():
        print(f"{stat}: {value}")

def make_norm_fig(args, norms_collection, epoch):
    colors = ['r', 'g', 'b', 'm']
    colors = colors[:len(norms_collection)]
    plt.figure(figsize=(10, 6))

    global_min = min(norms.min().item() for norms in norms_collection)
    global_max = max(norms.max().item() for norms in norms_collection)
    
    # Define the bins for the histogram
    bins = np.logspace(np.log10(global_min), np.log10(global_max), num=100)

    for i, norms in enumerate(norms_collection):
        print(norms.shape)
        counts, bin_edges = np.histogram(norms.cpu().numpy(), bins=100)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Plotting
        plt.semilogx(bin_centers, counts, label=f'Depth {i+1}', color=colors[i])

    plt.xlabel('Emb norms (log scale)')
    plt.ylabel('Density')
    plt.title('Codebook norms by depth')
    plt.legend()
    plt.grid(True)
    
    filename = f'norms_{epoch}.png'
    filepath = os.path.join(args.output_dir, filename)
    plt.savefig(filepath)
    plt.close()

def calc_code_stats(code_collection):
    stats = {'CUR': [], 'UE': []}
    for code_counts in code_collection:
        total_bins = code_counts.numel()
        non_zero_bins = (code_counts != 0).sum().item()
        
        # Calculate CUR
        CUR = non_zero_bins / total_bins
        stats['CUR'].append(CUR)
        
        # Calculate UE
        probabilities = code_counts[code_counts != 0].float() / code_counts.sum()
        log_probabilities = torch.log2(probabilities)
        UE = -torch.sum(probabilities * log_probabilities).item()
        stats['UE'].append(UE)
    
    return stats