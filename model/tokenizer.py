# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# LICENSE file in the root directory of this source tree.


from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible
from util.patch_embed import PatchEmbed_new, PatchEmbed_org
from .swin_transformer import SwinTransformerBlock
from .quantizations import NormEMAVectorQuantizer
from.rq_quantizations import RQBottleneck

class BEATsTokenizer(nn.Module):
    """ BEATs Tokenizer with Vision Transformer Encoder, Quantizer and Estimator
    """
    def __init__(self, img_size=224, patch_size=16, stride=10, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 estimator_embed_dim=768, estimator_depth=3, estimator_num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 audio_exp=False, mode=0, contextual_depth=8,
                 use_custom_patch=False, pos_trainable=False, estimator_mode=0,
                 codebook_type="legacy", code_num=1024, code_dim=256,
                 codebook_set=1, commitment_loss_weight=0.25, ema=True, 
                 epoch=0, no_shift=False,
                 # specifics for RQ codebook
                 restart_unused_codes=True, init_weight_multiplier=0.33,
                 ):
        super().__init__()

        self.audio_exp=audio_exp
        self.mode = mode
        self.embed_dim = embed_dim
        self.estimator_embed_dim = estimator_embed_dim
        # --------------------------------------------------------------------------
        # Tokenizer encoder specifics
        if use_custom_patch:
            print(f'Use custom patch_emb with patch size: {patch_size}, stride: {stride}')
            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        else:
            self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=False, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # codebook specifics
        self.cold_start_encoder = nn.Linear(embed_dim, code_dim, bias=False)
        self.quantize_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, code_dim)
        )
        if codebook_type == "legacy":
            self.quantizer = NormEMAVectorQuantizer(
                n_embed=code_num,
                embedding_dim=code_dim,
                commitment_loss_weight=commitment_loss_weight,
                kmeans_init=True,
                decay=0.99,
                ema=ema,
            )

            if not ema:
                self.quantizer.embedding.weight.requires_grad = True
        else:
            self.quantizer = RQBottleneck(
                latent_shape=[code_dim],
                code_shape=[codebook_set],
                n_embed=[code_num] * codebook_set,
                ema=ema,
                restart_unused_codes=restart_unused_codes,
                init_weight_multiplier=init_weight_multiplier,
                commitment_loss_weight=commitment_loss_weight,
            )

        if ema:
            for p in self.quantizer.parameters():
                p.requires_grad = False

        # --------------------------------------------------------------------------
        # Tokenizer estimator specifics
        self.estimator_embed = nn.Linear(embed_dim, estimator_embed_dim, bias=True)

        self.estimator_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, estimator_embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.no_shift=no_shift

        self.estimator_mode = estimator_mode
        if self.use_custom_patch: # overlapped patches as in AST. Similar performance yet compute heavy
            window_size= (6,6)
            input_size = (102,12)
        else:
            window_size= (4,4)
            input_size = (64,8)                
        if self.estimator_mode == 1:
            estimator_modules = []
            for index in range(16):
                if self.no_shift:
                    shift_size = (0, 0)
                else:
                    if (index % 2) == 0:
                        shift_size = (0, 0)
                    else:
                        shift_size = (2, 0)

                estimator_modules.append(
                    SwinTransformerBlock(
                        input_dim=estimator_embed_dim,
                        num_heads=16,
                        input_size=input_size,
                        window_size=window_size,
                        shift_size=shift_size,
                        feedforward_dim=mlp_ratio * estimator_embed_dim,
                    )
                )
            self.estimator_blocks = nn.ModuleList(estimator_modules)        
        else:
            # Transfomer
            self.estimator_blocks = nn.ModuleList([
                Block(estimator_embed_dim, estimator_num_heads, mlp_ratio, qkv_bias=True, qk_norm=False, norm_layer=norm_layer)
                for i in range(estimator_depth)])

        self.estimator_norm = norm_layer(estimator_embed_dim)
        self.estimator_pred = nn.Linear(estimator_embed_dim, patch_size**2 * in_chans, bias=True) # estimator to patch

        # --------------------------------------------------------------------------

        self.patch_size=patch_size
        self.stride=stride

        self.epoch = epoch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)    
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.audio_exp:   
            estimator_pos_embed = get_2d_sincos_pos_embed_flexible(self.estimator_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        else:
            estimator_pos_embed = get_2d_sincos_pos_embed(self.estimator_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.estimator_pos_embed.data.copy_(torch.from_numpy(estimator_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        L = (H/p)*(W/p)
        """
        p = self.patch_embed.patch_size[0]
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        if self.audio_exp:
            if self.use_custom_patch: # overlapped patch
                h,w = self.patch_embed.patch_hw
                # todo: fixed h/w patch size and stride size. Make hw custom in the future
                x = imgs.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride) # n,1,H,W -> n,1,h,w,p,p
                x = x.reshape(shape=(imgs.shape[0], h*w, p**2 * 1))
                #x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                #x = torch.einsum('nchpwq->nhwpqc', x)
                #x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            else:
                h = imgs.shape[2] // p
                w = imgs.shape[3] // p
                #h,w = self.patch_embed.patch_hw
                x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                x = torch.einsum('nchpwq->nhwpqc', x)
                x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else:
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        specs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]    
        h = 1024//p
        w = 128//p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return specs


    def forward_encoder(self, x, cold_start=False):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if cold_start:
            x = self.cold_start_encoder(x)
        else:
            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # apply Transformer blocks
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            # quantize w/o cls token
            x = x[:, 1:, :]

        return self.quantizer(x)

    def forward_estimator(self, x):
        # embed tokens
        x = self.estimator_embed(x)

        # x = x[:, 1:, :]  # no cls token

        # add pos embed
        x = x + self.estimator_pos_embed
        
        if self.estimator_mode != 0:
            B,L,D=x.shape
            x = x[:,1:,:]
            if self.use_custom_patch:
                x = x.reshape(B,101,12,D)
                x = torch.cat([x,x[:,-1,:].unsqueeze(1)],dim=1) # hack
                x = x.reshape(B,1224,D)
        if self.estimator_mode > 3: # mvit
            x = self.estimator_blocks(x)
        else:
            # apply Transformer blocks
            for blk in self.estimator_blocks:
                x = blk(x)
        x = self.estimator_norm(x)

        # predictor projection
        pred = self.estimator_pred(x)

        # remove cls token
        if self.estimator_mode != 0:
            if self.use_custom_patch:
                pred = pred.reshape(B,102,12,256)
                pred = pred[:,:101,:,:]
                pred = pred.reshape(B,1212,256)
            else:
                pred = pred
        else:
            pred = pred[:, 1:, :]
        return pred

    def forward(self, imgs, cold_start=False):
        _, _, enc_indices = self.forward_encoder(imgs, cold_start)
        return enc_indices

    def forward_estimator_embeddings(self, imgs):
        z_q, emb_loss, _ = self.forward_encoder(imgs)
        estimator_emb = self.forward_estimator(z_q)
        return estimator_emb, emb_loss
    

def tokenizer_vit_small_patch16(**kwargs):
    model = BEATsTokenizer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        estimator_embed_dim=384, estimator_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def tokenizer_vit_base_patch16(**kwargs):
    model = BEATsTokenizer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        estimator_embed_dim=768, estimator_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def tokenizer_vit_large_patch16(**kwargs):
    model = BEATsTokenizer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        estimator_embed_dim=1024, estimator_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
vit_b = tokenizer_vit_base_patch16
vit_l = tokenizer_vit_large_patch16
vit_s = tokenizer_vit_small_patch16