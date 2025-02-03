# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# LICENSE file in the root directory of this source tree.

# --------------------------------------------------------
# References:
# AudioMAE: https://github.com/facebookresearch/AudioMAE
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from util.stat import calculate_stats, concat_all_gather


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, 
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if args.precision == 'fp32':
        training_precision = torch.float32
    elif args.precision == 'fp16':
        training_precision = torch.float16
    else:
        raise ValueError(f"precision {args.precision} not supported")

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = batch[0]
        targets = batch[1]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.criterion == 'bce' and not args.audio_exp:
            # convert image and video labels to one hot
            targets = _get_one_hot_labels(targets, num_classes=args.nb_classes)
        elif args.criterion == 'ce' and args.audio_exp:
            # convert audio labels to class num
            targets = torch.argmax(targets, dim=-1)

        with torch.cuda.amp.autocast(dtype=training_precision):
            if args.dataset != 'k400':
                outputs = model(samples)
            else:
                outputs = model(samples, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)

            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_multilabel(data_loader, model, device, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    outputs=[]
    targets=[]

    if args.precision == 'fp32':
        training_precision = torch.float32
    elif args.precision == 'fp16':
        training_precision = torch.float16
    else:
        raise ValueError(f"precision {args.precision} not supported")

    for batch in metric_logger.log_every(data_loader, 300, header):

        images = batch[0]
        target = batch[1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(dtype=training_precision):
            if args.dataset != 'k400':
                outputs = model(images)
            else:
                outputs = model(images, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)

            if args.dist_eval:
                output = concat_all_gather(output)
                target = concat_all_gather(target)

            outputs.append(output)
            targets.append(target)

    outputs=torch.cat(outputs).cpu().numpy()
    targets=torch.cat(targets).cpu().numpy()

    stats = calculate_stats(outputs, targets)

    AP = [stat['AP'] for stat in stats]
    mAP = np.mean([stat['AP'] for stat in stats])
    print("mAP: {:.6f}".format(mAP))
    return {"mAP": mAP, "AP": AP}


@torch.no_grad()
def evaluate_single_label(data_loader, model, device, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if args.precision == 'fp32':
        training_precision = torch.float32
    elif args.precision == 'fp16':
        training_precision = torch.float16
    else:
        raise ValueError(f"precision {args.precision} not supported")

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.audio_exp:
            targets = torch.argmax(targets, dim=-1)

        # compute output
        with torch.cuda.amp.autocast(dtype=training_precision):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def _get_one_hot_labels(labels, num_classes):
    one_hot = torch.zeros(labels.size() + (num_classes,), device=labels.device)
    return one_hot.scatter(-1, labels[..., None], 1).squeeze(-1)