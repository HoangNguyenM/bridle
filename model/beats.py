# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# LICENSE file in the root directory of this source tree.


import torch
from torch import nn, Tensor


class BEATs(nn.Module):
    """ BEATs model with the following component: 
        encoder - decoder: backbone is a masked auto encoder
        tokenizer encoder: ViT, equivalent structure to main encoder
        quantizer (codebook): VQ or RQ embeddings
        tokenizer estimator: simple transformer
    """
    def __init__(self,
                 encoder_decoder: nn.Module,
                 tokenizer: nn.Module,
                 cold_start: bool = False,
                 train_encoder: bool = True,
                 model_loss: nn.Module = nn.CrossEntropyLoss(),
                 codebook_type: str = "legacy",
                 dual_train: bool = False,
                 ):
        super().__init__()

        self.encoder_decoder = encoder_decoder
        self.tokenizer = tokenizer
        self.cold_start = cold_start
        self.train_encoder = train_encoder
        self.codebook_type = codebook_type
        self.model_loss = model_loss

        if dual_train:
            self.encoder_decoder.train(True)
            self.tokenizer.train(True)
        elif train_encoder:
            self.encoder_decoder.train(True)
            self.tokenizer.train(False)
            self.freeze_model(self.tokenizer)
        else:
            self.encoder_decoder.train(False)
            self.tokenizer.train(True)
            self.freeze_model(self.encoder_decoder)

    def freeze_model(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def _get_one_hot_labels(self, labels: Tensor, num_classes) -> Tensor:
        one_hot = torch.zeros(labels.size() + (num_classes,), device=labels.device)
        return one_hot.scatter(-1, labels[..., None], 1).squeeze(-1)

    def cosine_similarity_loss(self, encoder_output, estimator_output):
        if encoder_output.shape[1] - estimator_output.shape[1] == 1:
            encoder_output = encoder_output[:, 1:, :]
        target = encoder_output / encoder_output.norm(dim=-1, keepdim=True)
        pred = estimator_output / estimator_output.norm(dim=-1, keepdim=True)
        loss = (1 - (target * pred).sum(-1)).mean()
        return loss
    
    def encoder_training_loss(self, 
            decoder_pred=None,
            quantized_tokens=None,
            mask=None):

        total_code_num = decoder_pred.shape[-1]
        # apply masking here for Cross Entropy loss
        mask_indices = mask.bool()
        decoder_pred = decoder_pred[
            mask_indices.unsqueeze(-1).repeat(1, 1, total_code_num)
        ]

        if self.codebook_type == "rq":
            # for RQ, total_code_num = code_num * codebook_set
            code_num = total_code_num // quantized_tokens.shape[-1]
            codebook_set = quantized_tokens.shape[-1]
            quantized_tokens = quantized_tokens[
                mask_indices.unsqueeze(-1).repeat(1, 1, codebook_set)
            ].view(-1)
        else:
            code_num = total_code_num
            quantized_tokens = quantized_tokens[mask_indices].view(-1)
        decoder_pred = decoder_pred.view(-1, code_num)

        # Use Cross Entropy loss for training encoder_decoder
        if isinstance(self.model_loss, nn.BCEWithLogitsLoss):
            one_hot_tokens = self._get_one_hot_labels(
                quantized_tokens, code_num
            )
            return self.model_loss(decoder_pred, one_hot_tokens)
        else:
            return self.model_loss(decoder_pred, quantized_tokens)

    
    def tokenizer_training_loss(self, 
            encoder_output=None,
            estimator_output=None,
            embed_loss=None):
        
        # Use Cosine Similarity loss for training tokenizer and mse loss for the codebook
        return self.cosine_similarity_loss(encoder_output, estimator_output) + embed_loss

    def forward_encoder_training(self, x, mask_ratio=0.8):
        pred, mask = self.encoder_decoder(x, mask_ratio)
        with torch.no_grad():
            enc_indices = self.tokenizer(x, self.cold_start)
        
        loss = self.encoder_training_loss(pred, enc_indices, mask)
        return loss

    def forward_tokenizer_training(self, x):
        with torch.no_grad():
            emb_enc = self.encoder_decoder(x, 0.0, encoder_training=False)
        estimator_emb, emb_loss = self.tokenizer.forward_estimator_embeddings(x)

        loss = self.tokenizer_training_loss(emb_enc, estimator_emb, emb_loss)
        return loss
    
    def forward(self, x, mask_ratio=0.8):
        if self.train_encoder:
            return self.forward_encoder_training(x, mask_ratio)
        else:
            return self.forward_tokenizer_training(x)
        

class BEATsDualTrain(BEATs):
    """ BEATs upgraded to train both the encoder decoder and tokenizer simultaneously
    """
    def __init__(self,
                 encoder_decoder: nn.Module,
                 tokenizer: nn.Module,
                 model_loss: nn.Module = nn.CrossEntropyLoss(),
                 codebook_type: str = "legacy",
                 tokenizer_loss_ratio: float = 0.1,
                 ):
        super().__init__(
            encoder_decoder=encoder_decoder,
            tokenizer=tokenizer,
            cold_start=False,
            train_encoder=True,
            model_loss=model_loss,
            codebook_type=codebook_type,
            dual_train=True,
        )

        self.tokenizer_loss_ratio = tokenizer_loss_ratio

    def forward(self, x, mask_ratio=0.8):
        encoder_loss = self.forward_encoder_training(x, mask_ratio)
        tokenizer_loss = self.forward_tokenizer_training(x)
        return encoder_loss + self.tokenizer_loss_ratio * tokenizer_loss