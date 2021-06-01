import logging
import pytorch_lightning as pl
import torch.nn as nn
import torchio as tio

from torch.optim import Optimizer

class LightningUnet(pl.LightningModule):
    """ Lightning model for automation of unet trainingg

    Args:
        - criterion (nn.functional): loss function
        - optimizer_class (torch.optim.Optimizer):
        - unet (nn.Module): model to use.
        - learning rate (float): learning rate. Defaults to 0.1
    """

    def __init__(self,
                # Training arguments
                criterion: nn.Module, 
                optimizer_class: Optimizer,
                unet: nn.Module,
                learning_rate: float = 0.1
                ):
        super(LightningUnet, self).__init__()

        
        self.help_logger = logging.getLogger(__name__)
        self.lr = learning_rate
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.unet = unet
        self.is_group = self.unet.group

        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
    
    def prepare_batch(self, batch):
        if self.is_group:
            image = batch['image'][tio.DATA].unsqueeze(1)
            return image, batch['label'][tio.DATA]
        else:
            return batch['image'][tio.DATA], batch['label'][tio.DATA]
    
    def infer_batch(self, batch):
        inputs, labels = self.prepare_batch(batch)
        outputs = self.forward(inputs)
        return outputs, labels

    def training_step(self, batch, batch_idx):
        outputs, targets = self.infer_batch(batch)
        _, targets = targets.max(1) # Voir avec Clément pourquoi on a besoin de ça déjà
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs, targets = self.infer_batch(batch)
        _, targets = targets.max(1)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss
 