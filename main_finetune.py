# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# LICENSE file in the root directory of this source tree.

# --------------------------------------------------------
# References:
# AudioMAE: https://github.com/facebookresearch/AudioMAE
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models.layers import to_2tuple

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util_video.utils import multiple_samples_collate

import model.vit as vit
import model_video.vvit as vvit

from unit_finetune import train_one_epoch

from dataset import AudiosetDataset, DistributedWeightedSampler, DistributedSamplerWrapper

from torch.utils.data import WeightedRandomSampler

def get_args_parser():
    parser = argparse.ArgumentParser('Finetuning', add_help=False)
    parser.add_argument('--linear_probe', action='store_true')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_b', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT', help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--save_every_epoch', default=20, type=int,help='save_every_epoch')
    parser.add_argument('--save_min_epoch', default=60, type=int,help='save_min_epoch')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', 
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',help='epochs to warmup LR')
    parser.add_argument('--precision', default='fp16', type=str, help='training precision')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--num_sample', type=int, default=2, help='Repeated_aug (default: 2)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, 
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.5, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str, help='dataset path')
    parser.add_argument('--nb_classes', default=527, type=int, help='number of the classification types')

    parser.add_argument('--output_dir', default='/checkpoints/hoangmn/finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/checkpoints/hoangmn/finetune',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--distributed", type=bool, default=True)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # For audioset
    parser.add_argument('--audio_exp', action='store_false', help='audio exp')
    parser.add_argument("--data_train", type=str, default='/fsx/hoangmn/audioset/train_all.json', help="training data json")
    parser.add_argument("--data_eval", type=str, default='/fsx/hoangmn/audioset/eval.json', help="validation data json")    
    parser.add_argument("--label_csv", type=str, default='/fsx/hoangmn/audioset/class_labels_indices.csv', help="csv with class labels")
    parser.add_argument("--weight_csv", type=str, default='/fsx/hoangmn/audioset/weight_train_all.csv', help="weight file")
    
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=192)
    parser.add_argument('--timem', help='time mask max length', type=int, default=48)
    #parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "k400"])
    parser.add_argument("--use_fbank", type=bool, default=False)
    parser.add_argument("--use_soft", type=bool, default=False)
    parser.add_argument("--fbank_dir", type=str, default="/fsx/hoangmn/esc_50/ESC-50-master/fbank", help="fbank dir") 

    parser.add_argument('--first_eval_ep', default=0, type=int, help='do eval after first_eval_ep')
    parser.add_argument('--source_custom_patch', type=bool, default=False, help='the pre-trained model already use custom patch')
    parser.add_argument('--roll_mag_aug', type=bool, default=False, help='use roll_mag_aug')
    parser.add_argument('--mask_t_prob', default=0.0, type=float, help='T masking ratio (percentage of removed patches).') #  
    parser.add_argument('--mask_f_prob', default=0.0, type=float, help='F masking ratio (percentage of removed patches).') #  
    #parser.add_argument('--split_pos', type=bool, default=False, help='use splitted pos emb')
    parser.add_argument('--weight_sampler', action='store_true', help='use weight_sampler')
    parser.add_argument('--epoch_len', default=200000, type=int, help='num of samples/epoch with weight_sampler')
    parser.add_argument('--distributed_wrapper', type=bool, default=False, help='use distributedwrapper for weighted sampler')
    parser.add_argument('--replacement', action='store_true', help='use weight_sampler')
    parser.add_argument('--mask_2d', type=bool, default=False, help='use 2d masking')
    parser.add_argument('--load_video', type=bool, default=False, help='load video')
    parser.add_argument('--av_fusion', type=bool, default=False, help='load video')
    parser.add_argument('--n_frm', default=6, type=int, help='num of frames for video')
    parser.add_argument('--replace_with_mae', type=bool, default=False, help='replace_with_mae')
    parser.add_argument('--load_imgnet_pt', type=bool, default=False, help='when img_pt_ckpt, if load_imgnet_pt, use img_pt_ckpt to initialize audio branch, if not, keep audio branch random')
    
    # For video dataset (Kinetics)
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--patch_size', type=int, default= 16)
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--new_height', type=int, default= 256)
    parser.add_argument('--new_width', type=int, default= 320)

    parser.add_argument('--num_segments', type=int, default= 1)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)

    return parser


class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # audio dataset
    if args.audio_exp:
        norm_stats = {'audioset':[-4.4446096, 3.3216383], 'k400':[-4.2677393, 4.5689974], 
                      'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
        
        target_length = {'audioset':1024, 'k400':1024, 'esc50':512, 'speechcommands':128}
        multilabel_dataset = {'audioset': True, 'esc50': False, 'k400': False, 'speechcommands': True}
        audio_conf_train = {'num_mel_bins': 128, 
                      'target_length': target_length[args.dataset], 
                      'freqm': 48,
                      'timem': 192,
                      'mixup': args.mixup,
                      'dataset': args.dataset,
                      'mode':'train',
                      'mean':norm_stats[args.dataset][0],
                      'std':norm_stats[args.dataset][1],
                      'noise':False,
                      'multilabel':multilabel_dataset[args.dataset],
                      }
        audio_conf_val = {'num_mel_bins': 128, 
                      'target_length': target_length[args.dataset], 
                      'freqm': 0,
                      'timem': 0,
                      'mixup': 0,
                      'dataset': args.dataset,
                      'mode':'val',
                      'mean':norm_stats[args.dataset][0],
                      'std':norm_stats[args.dataset][1],
                      'noise':False,
                      'multilabel':multilabel_dataset[args.dataset],
                      }  
        dataset_train = AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf_train, 
                                        use_fbank=args.use_fbank, fbank_dir=args.fbank_dir, 
                                        roll_mag_aug=args.roll_mag_aug, load_video=args.load_video, mode='train', dataset=args.dataset)
        dataset_val = AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=audio_conf_val, 
                                        use_fbank=args.use_fbank, fbank_dir=args.fbank_dir, 
                                        roll_mag_aug=False, load_video=args.load_video, mode='eval', dataset=args.dataset)
        args.multilabel = multilabel_dataset[args.dataset]
    # video dataset
    elif args.dataset == 'k400':
        args.window_size = (args.num_frames // 2, args.input_size // args.patch_size, args.input_size // args.patch_size)
        dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
        dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
        args.multilabel = False
    # image dataset
    else:
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)
        args.multilabel = False

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        num_nodes = int(os.environ.get('num_nodes', 1))
        ddp = int(os.environ.get('DDP', 1))
        num_nodes = max(ddp, num_nodes)
        rank = int(os.environ.get('NODE_RANK', 0))
        print(f"num_nodes:{num_nodes}, rank:{rank}, ddp:{ddp}, num_tasks:{num_tasks}, global_rank:{global_rank}")
        # num_nodes:1, rank:0, ddp:1, num_tasks:8, global_rank:0 (sbatch)
        if args.weight_sampler:
            samples_weight = np.loadtxt(args.weight_csv, delimiter=',')
            if args.distributed_wrapper:
                print('use distributed_wrapper sampler')
                epoch_len=args.epoch_len #200000 #=> 250000
                #epoch_len=21000 # AS-20K
                # replacement should be False
                sampler_train = DistributedSamplerWrapper(
                                    sampler=WeightedRandomSampler(samples_weight, num_samples=epoch_len, 
                                    replacement=args.replacement),
                                    dataset=range(epoch_len),
                                    num_replicas=num_tasks, #num_nodes, #num_tasks?
                                    rank=global_rank, #rank, # global_rank?
                                    )
            else:
                #sampler_train = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
                sampler_train = DistributedWeightedSampler(dataset_train, samples_weight,  
                                    num_replicas=num_tasks, rank=global_rank, replacement=args.replacement)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    print(log_writer)
    
    if args.dataset == 'k400' and args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    if args.dataset == 'k400':
        model = vvit.__dict__[args.model](
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            fc_drop_rate=args.fc_drop_rate,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
    )
    else:
        model = vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            mask_2d=args.mask_2d,
        )

    if args.audio_exp:
        img_size=(target_length[args.dataset],128) # 1024, 128
        in_chans=1
        emb_dim = 768
        if args.model == "vit_s":
            emb_dim = 384

        model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
        num_patches = model.patch_embed.num_patches
        #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        # modify checkpoint keys from pretrained checkpoint
        checkpoint_model = {k.replace("encoder_decoder.", "", 1): v 
                            for k, v in checkpoint_model.items() if "encoder_decoder" in k}

        if args.dataset == 'esc50':
            embed_weights = checkpoint_model['pos_embed']
            new_embed_weights = embed_weights[:, 128:384, :]
            checkpoint_model['pos_embed'] = torch.cat([embed_weights[:, :1, :], new_embed_weights], dim=1)
        elif args.dataset == 'k400':
            if 'pos_embed' in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
                num_patches = model.patch_embed.num_patches # 
                num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

                # height (== width) for the checkpoint position embedding 
                orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    # B, L, C -> BT, H, W, C -> BT, C, H, W
                    pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                    pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model['pos_embed'] = new_pos_embed

        if not args.eval:
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        if not args.eval:
            trunc_normal_(model.head.weight, std=2e-5)

    # freeze all layers except head & pooling if linear probe
    if args.linear_probe:
        for name, p in model.named_parameters():
            if name not in ['head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias']:
                p.requires_grad = False

    for name, p in model.named_parameters():
            print(f"{name}: requires_grad = {p.requires_grad}")

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    loss_scaler = NativeScaler()

    if args.use_soft:
        criterion = SoftTargetCrossEntropy()
    elif args.criterion == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    print("criterion = %s" % str(criterion))

    if args.multilabel:
        from unit_finetune import evaluate_multilabel as evaluate
    else:
        from unit_finetune import evaluate_single_label as evaluate

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        # currently only support AudioSet
        test_stats = evaluate(data_loader_val, model, device, args.dist_eval)
        with open('aps.txt', 'w') as fp:
            aps=test_stats['AP']
            aps=[str(ap) for ap in aps]
            fp.write('\n'.join(aps))
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['mAP']:.4f}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_mAP = 0.0
    max_acc1 = 0.0
    max_acc5 = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn, log_writer=log_writer,
                args=args
            )
        
        # uncomment to save checkpoints during finetuning/linear probing
        if args.output_dir and epoch > args.save_min_epoch \
                and (epoch % args.save_every_epoch == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if epoch >= args.first_eval_ep:
            test_stats = evaluate(data_loader_val, model, device, args=args)
            if args.multilabel:
                print(f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']:.4f}")
                max_mAP = max(max_mAP, test_stats["mAP"])
                print(f'Max mAP: {max_mAP:.4f}')
            else:
                print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.4f}")
                max_acc1 = max(max_acc1, test_stats["acc1"])
                max_acc5 = max(max_acc5, test_stats["acc5"])
                print(f'Max accuracy top1: {max_acc1:.4f}, max accuracy top5: {max_acc5:.4f}')
        else:
            if args.multilabel:
                test_stats = {'mAP': 0.0}
            else:
                test_stats = {'acc1': 0.0, 'acc5': 0.0}
            print(f'too new to evaluate!')

        if log_writer is not None:
            log_writer.add_scalar('perf/mAP', test_stats['mAP'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)