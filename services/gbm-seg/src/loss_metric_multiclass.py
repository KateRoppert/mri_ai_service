from typing import List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from utils import to_tensor
import numpy as np
from loss_metric import dice_coef_metric_batch_fair, dice_coef_metric_3d

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"

def binary_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold=None,
    nan_score_on_empty=False,
    eps=1e-7,
) -> float:
    """
    Compute IoU score between two image tensors
    :param y_pred: Input image tensor of any shape
    :param y_true: Target image of any shape (must match size of y_pred)
    :param mode: Metric to compute (dice, iou)
    :param threshold: Optional binarization threshold to apply on @y_pred
    :param nan_score_on_empty: If true, return np.nan if target has no positive pixels;
        If false, return 1. if both target and input are empty, and 0 otherwise.
    :param eps: Small value to add to denominator for numerical stability
    :return: Float scalar
    """
    assert mode in {"dice", "iou"}

    # Binarize predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if mode == "dice":
        score = (2.0 * intersection) / (cardinality + eps)
    else:
        score = intersection / (cardinality + eps)

    has_targets = torch.sum(y_true) > 0
    has_predicted = torch.sum(y_pred) > 0

    if not has_targets:
        if nan_score_on_empty:
            score = np.nan
        else:
            score = float(not has_predicted)
    return score
    

def soft_dice_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, smooth=1e-7, eps=1e-7, dims=None
) -> torch.Tensor:
    """
    :param y_pred:
    :param y_true:
    :param smooth:
    :param eps:
    :return:
    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.
    """
    assert y_pred.size() == y_true.size()
    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        union = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred + y_true)
    dice_score = (2.0 * intersection + smooth) / (union + smooth + eps)
    return dice_score


class DiceLoss_multi(_Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss=False,
        from_logits=True,
        smooth=1e-7,
        eps=1e-7,
    ):
        """
        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation; By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss_multi, self).__init__()
        self.mode = mode
        if classes is not None:
            assert (
                mode != BINARY_MODE
            ), "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.softmax(dim=1)
            else:
                y_pred = y_pred.sigmoid()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = soft_dice_loss(
            y_pred, y_true.type(y_pred.dtype), self.smooth, self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores)
        else:
            loss = 1 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        # mask = (y_true.sum(dims) > 0).float()
        # loss = loss * mask

        if self.classes is not None:
            loss = loss[self.classes]

        # print (f'MULTICLASS DICE: {loss}')

        return loss.mean()


def bce_dice_loss_multiclass(input, target, weight=0.5, loss_classes=None, ce_weights=None, logits=False):
    DICEcriterion = DiceLoss_multi(mode="multiclass", classes=loss_classes, from_logits=logits)
    DICEloss = DICEcriterion(input, target)
    CEcriterion = nn.CrossEntropyLoss(weight=ce_weights).cuda()
    CEloss = CEcriterion(input, target)
    return weight*CEloss + (1-weight)*DICEloss, CEloss, DICEloss



def multiclass_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
    classes_of_interest=None,
):
    ious = []
    num_classes = y_pred.size(0)
    y_pred = y_pred.argmax(dim=0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        iou = binary_dice_iou_score(
            y_pred=(y_pred == class_index).float(),
            y_true=(y_true == class_index).float(),
            mode=mode,
            nan_score_on_empty=nan_score_on_empty,
            threshold=threshold,
            eps=eps,
        )
        ious.append(iou)

    # print (f'MULTICLASS IOUS: {ious}')

    return ious



def multi_dice_coef_metric_batch_fair(outputs, targets, num_classes):
    dices_class = []
    for class_index in range(1,num_classes+1):
        dice_class = dice_coef_metric_batch_fair((outputs == class_index), (targets == class_index))
        # print(dice_class)
        dices_class.append(dice_class)

    return dices_class

def multi_dice_coef_metric_batch_fair_3d(outputs, targets, num_classes):
    dices = []
    for ij in range(1,num_classes+1):
        dices.append(dice_coef_metric_3d(outputs==ij,targets==ij))
    return dices

