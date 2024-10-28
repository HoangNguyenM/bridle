# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# LICENSE file in the root directory of this source tree.

# --------------------------------------------------------
# References:
# AST: https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py
# --------------------------------------------------------

import torch
from torch import nn

import timm
from timm.models.layers import to_2tuple


def get_imagenet_pretrained_model(model_type='vit_b', img_size=(1024, 128), patch_size=16):
    assert model_type == 'vit_b'
    model = timm.create_model('vit_base_patch16_224', pretrained=True)

    original_num_patches = model.patch_embed.num_patches
    oringal_hw = int(original_num_patches ** 0.5)
    original_embedding_dim = model.pos_embed.shape[2]

    # automatcially get the intermediate shape
    img_size = to_2tuple(img_size)
    patch_size = to_2tuple(patch_size)
    f_dim, t_dim = img_size[1] // patch_size[1], img_size[0] // patch_size[0]
    num_patches = f_dim * t_dim
    model.patch_embed.num_patches = num_patches

    # the linear projection layer
    new_proj = torch.nn.Conv2d(1, original_embedding_dim, kernel_size=(16, 16), stride=(16, 16))
    new_proj.weight = torch.nn.Parameter(torch.sum(model.patch_embed.proj.weight, dim=1).unsqueeze(1))
    new_proj.bias = model.patch_embed.proj.bias
    model.patch_embed.proj = new_proj

    # adjust for imagenet
    # get the positional embedding from model, skip the first token (cls token), reshape it to original 2D shape (24*24).
    new_pos_embed = model.pos_embed[:, 1:, :].detach().reshape(1, original_num_patches, original_embedding_dim).transpose(1, 2).reshape(1, original_embedding_dim, oringal_hw, oringal_hw)
    # cut (from middle) or interpolate the second dimension of the positional embedding
    if t_dim <= oringal_hw:
        new_pos_embed = new_pos_embed[:, :, :, int(oringal_hw / 2) - int(t_dim / 2): int(oringal_hw / 2) - int(t_dim / 2) + t_dim]
    else:
        new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(oringal_hw, t_dim), mode='bilinear')
    # cut (from middle) or interpolate the first dimension of the positional embedding
    if f_dim <= oringal_hw:
        new_pos_embed = new_pos_embed[:, :, int(oringal_hw / 2) - int(f_dim / 2): int(oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
    else:
        new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
    # flatten the positional embedding
    new_pos_embed = new_pos_embed.reshape(1, original_embedding_dim, num_patches).transpose(1,2)
    # concatenate the above positional embedding with the cls token of the model.
    model.pos_embed = nn.Parameter(torch.cat([model.pos_embed[:, :1, :].detach(), new_pos_embed], dim=1))

    return model