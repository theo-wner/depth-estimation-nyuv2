import torch
import torchmetrics
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import math
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import config
from utils import PolyLR
from utils import RMSLELoss
from utils import compute_depth_metrics

"""
Defines the model
"""

class DepthFormer(pl.LightningModule):

    def __init__(self):

        super().__init__()

        pl.seed_everything(42)       # reproducibility

        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, semantic_loss_ignore_index = config.IGNORE_INDEX, return_dict=False)
        self.model = SegformerForSemanticSegmentation(model_config) # this does not load the imagenet weights yet.
        self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)  # this loads the weights

        # Initialize the loss function
        self.loss_fn = RMSLELoss()

        # Initialize the metrics
        self.metrics = compute_depth_metrics

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)


    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_index):
        images, depths = batch

        _, preds = self.model(images, depths.squeeze(dim=1)) # _ because the model computes the loss internally but we compute it manually below, depths are also merely needed as dummies
        
        preds = torch.nn.functional.interpolate(preds, size=images.shape[-2:], mode="bilinear", align_corners=False)    # upsample preds to input image size (SegFormer outputs h/4 and w/4 by default, see paper)

        preds = F.relu(preds, inplace=True)
        
        valid_mask = (depths > 1e-3) & (depths < 10)    # common practice: https://arxiv.org/pdf/2207.04535v2.pdf
        
        loss = self.loss_fn(preds, depths, valid_mask)
        
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_index):
        images, depths = batch

        _, preds = self.model(images, depths.squeeze(dim=1)) # _ because the model computes the loss internally but we compute it manually below, depths are also merely needed as dummies
        
        preds = torch.nn.functional.interpolate(preds, size=images.shape[-2:], mode="bilinear", align_corners=False)    # upsample preds to input image size (SegFormer outputs h/4 and w/4 by default, see paper)

        preds = F.relu(preds, inplace=True)

        valid_mask = (depths > 1e-3) & (depths < 10)

        preds = preds[valid_mask]
        depths = depths[valid_mask]

        metrics = self.metrics(preds, depths)
        
        self.log('val_rmse', metrics[0], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_abs_rel', metrics[1], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_log10', metrics[2], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_d1', metrics[3], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_d2', metrics[4], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_d3', metrics[5], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)