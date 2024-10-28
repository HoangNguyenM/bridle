# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# LICENSE file in the root directory of this source tree.

from typing import Iterable
import os

import matplotlib.pyplot as plt
import numpy as np
import math

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

    for data_iter_step, (samples, _labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)

        if args.precision == 'fp32':
            training_precision = torch.float32
        elif args.precision == 'fp16':
            training_precision = torch.float16
        else:
            raise ValueError(f"precision {args.precision} not supported")

        with torch.cuda.amp.autocast(dtype=training_precision):
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
        # all_norms_global = [torch.zeros_like(all_norms) for _ in range(dist.get_world_size())]
        # dist.all_gather(all_norms_global, all_norms)
        # all_norms_global = torch.cat(all_norms_global)
        # norms_collection.append(all_norms_global)
        norms_collection.append(all_norms)
    
    for i in range(args.codebook_set):
        dist.all_reduce(code_collection[i], op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        make_norm_fig(args, norms_collection, epoch)

    stats = calc_code_stats(code_collection)
    for stat, value in stats.items():
        print(f"{stat}: {value}")

def make_norm_fig(args, norms_collection, epoch):
    colors = ['b', 'g', 'r', 'm']
    colors = colors[:len(norms_collection)]
    plt.figure(figsize=(10, 6))

    global_min = min(norms.min().item() for norms in norms_collection)
    global_max = max(norms.max().item() for norms in norms_collection)
    
    # Define the bins for the histogram
    # bins = np.logspace(np.log10(global_min), np.log10(global_max), num=100)
    bins = np.linspace(global_min, global_max, num=100)

    for i, norms in enumerate(norms_collection):
        print(norms.shape)
        counts, bin_edges = np.histogram(norms.cpu().numpy(), bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Print top 10 bins for the current tensor
        top_bins_indices = np.argsort(counts)[-10:]
        top_bins_counts = counts[top_bins_indices]
        for j, count in enumerate(top_bins_counts, 1):
            print(f"  Bin {j}: Count = {count}, Range = ({bin_edges[top_bins_indices[j-1]]}, {bin_edges[top_bins_indices[j-1]+1]})")
        # Plotting
        # plt.semilogx(bin_centers, counts, label=f'Depth {i+1}', color=colors[i])
        plt.hist(bin_centers, bins=bins, weights=counts, density=True, label=f'Depth {i+1}', color=colors[i])

    plt.xlabel('Embeddings norms')
    plt.ylabel('Counts (log scale)')
    plt.title('Codebook norms by depth')
    plt.legend()
    plt.grid(True)

    plt.yscale('log')
    # plt.xscale('log')
    
    filename = f'norms_{epoch}.png'
    filepath = os.path.join(args.output_dir, filename)
    plt.savefig(filepath)
    plt.close()

def calc_code_stats(code_collection):
    stats = {'CUR': [], 'UE': [], 'ECU': []}
    for code_counts in code_collection:
        print(code_counts.sum())
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

        # Calculate ECU
        if UE != 0:
            ECU = CUR / (UE * math.log(total_bins))
        else:
            ECU = float('inf')
        stats['ECU'].append(ECU)
    
    return stats