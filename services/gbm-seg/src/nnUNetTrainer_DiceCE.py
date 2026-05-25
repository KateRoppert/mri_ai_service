from monai.losses import DiceCELoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
import torch


class CorrectedDiceCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, pos_w):
        super().__init__()

        self.dice_ce_loss = DiceCELoss(to_onehot_y=False, sigmoid=True)

    def forward(self, y_hat, y):
        y_hat = y_hat[:, 1:, ...]
        return self.dice_ce_loss(y_hat, y)


class nnUNetTrainer_DiceCE(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        self.initial_lr = 1e-2
        self.num_epochs = 500
        self.save_every = 5
        self.disable_checkpointing = False
    
    def _build_loss(self):
        loss = CorrectedDiceCELoss(0.3, 0.7, 10)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)

        return loss
