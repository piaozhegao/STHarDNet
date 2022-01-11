import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import numpy as np
import logging
# PyTorch
ALPHA = 0.8
GAMMA = 2


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class DiceFocal(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceFocal, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        BCE = F.cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        dice = tgm.losses.DiceLoss()
        dice_loss = dice(inputs, targets)

        # print(focal_loss, dice_loss)
        return focal_loss + dice_loss


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)
        #
        # # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        dice = tgm.losses.DiceLoss()
        dice_loss = dice(inputs, targets)
        loss_final = dice_loss
        return loss_final


# PyTorch
class DiceCE_cherhoo(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCE_cherhoo, self).__init__()

    def iou_loss(self, logits,  true, eps=1e-7):
        """Computes the Jaccard loss, a.k.a the IoU loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the jaccard loss so we
        return the negated jaccard loss.
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            jacc_loss: the Jaccard loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return (1 - jacc_loss)

    def dice_loss(self, logits,true, eps=1e-7):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)].type(torch.long)
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)

    def cross_entorpy(self, logits, true):
        num_classes = logits.shape[1]
        # logits = F.sigmoid(logits)
        # # flatten label and prediction tensors
        # logits = logits.view(-1)
        # true = true.view(-1)

        loss_ce = 0
        if num_classes == 1:
            loss_ce = F.binary_cross_entropy(logits, true, reduction='mean')
        else:
            loss_ce = F.cross_entropy(logits, true, reduction='mean')
        return loss_ce

    def forward(self, inputs, targets, smooth=1):
        ## iou loss
        #my_iou_loss = self.iou_loss(inputs, targets)

        ## dice loss
        my_dice_loss = self.dice_loss(inputs, targets)
        ## cross entropy
        my_cross_entorpy = self.cross_entorpy(inputs, targets)

        final_loss = my_dice_loss + my_cross_entorpy
        ###  iou*0.2  dice*0.4  cross entropy * 0.4
        #final_loss = 0.2*my_iou_loss + 0.4*my_dice_loss + 0.4*my_cross_entorpy
        return final_loss







# PyTorch
class DiceCE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCE, self).__init__()

    def forward(self, inputs, targets):
        # class_weights = [2.422, 5.675, 2.276, 0.428, 0.871, 0.671]
        # class_weights = torch.FloatTensor(class_weights).cuda()
        # ce_loss = F.cross_entropy(inputs, targets, weight=class_weights, reduction='mean')
        ce_loss = F.cross_entropy(inputs, targets, reduction='mean')

        dice = tgm.losses.DiceLoss()
        dice_loss = dice(inputs, targets)
        #loss_final = 0.4*ce_loss + 0.6*dice_loss
        loss_final = ce_loss + dice_loss
        return loss_final

class DiceCE_multi(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCE_multi, self).__init__()

    def forward(self, inputs, inputs_cnn, inputs_swin, targets):
        # label_smoothing_ce = LabelSmoothingCrossEntropy(reduction='mean')
        # ce_loss = label_smoothing_ce(inputs, targets)

        # loss = (0.4 * F.cross_entropy(inputs, targets, reduction='mean')) + (0.6 * tgm.losses.DiceLoss()(inputs, targets))
        # loss_cnn = (0.4 * F.cross_entropy(inputs_cnn, targets, reduction='mean')) + (0.6 * tgm.losses.DiceLoss()(inputs, targets))
        # loss_swin = (0.4 * F.cross_entropy(inputs_swin, targets, reduction='mean')) + (0.6 * tgm.losses.DiceLoss()(inputs, targets))
        loss = (1 * F.cross_entropy(inputs, targets, reduction='mean')) + (1 * tgm.losses.DiceLoss()(inputs, targets))
        loss_cnn = (1 * F.cross_entropy(inputs_cnn, targets, reduction='mean')) + (1 * tgm.losses.DiceLoss()(inputs, targets))
        loss_swin = (1 * F.cross_entropy(inputs_swin, targets, reduction='mean')) + (1 * tgm.losses.DiceLoss()(inputs, targets))
        final_loss = 0.6*loss + 0.2*loss_cnn + 0.2*loss_swin
        return final_loss


# PyTorch
class Dice_IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice_IoULoss, self).__init__()

    def iou_loss(self, logits,  true, eps=1e-7):
        """Computes the Jaccard loss, a.k.a the IoU loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the jaccard loss so we
        return the negated jaccard loss.
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            jacc_loss: the Jaccard loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return (1 - jacc_loss)

    def dice_loss(self, logits,true, eps=1e-7):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)].type(torch.long)
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)

    def cross_entorpy(self, logits, true):
        num_classes = logits.shape[1]
        # logits = F.sigmoid(logits)
        # # flatten label and prediction tensors
        # logits = logits.view(-1)
        # true = true.view(-1)

        loss_ce = 0
        if num_classes == 1:
            loss_ce = F.binary_cross_entropy(logits, true, reduction='mean')
        else:
            loss_ce = F.cross_entropy(logits, true, reduction='mean')
        return loss_ce

    def forward(self, inputs, targets, smooth=1):
        ## iou loss
        #my_iou_loss = self.iou_loss(inputs, targets)

        ## dice loss
        my_dice_loss = self.dice_loss(inputs, targets)
        ## cross entropy
        #my_cross_entorpy = self.cross_entorpy(inputs, targets)

        final_loss = my_dice_loss
        ###  iou*0.2  dice*0.4  cross entropy * 0.4
        #final_loss = 0.2*my_iou_loss + 0.4*my_dice_loss + 0.4*my_cross_entorpy
        return final_loss



# PyTorch
class Dice_IoULoss_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice_IoULoss_binary, self).__init__()

    def iou_loss(self, inputs, targets, smooth=1e-6):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


    def dice_loss(self, inputs, targets, smooth=1e-6):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


    def cross_entorpy(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        loss_ce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return loss_ce

    def forward(self, inputs, targets, smooth=1):
        #inputs = inputs.long()
        #targets = targets.long()
        ## iou loss
        my_iou_loss = self.iou_loss(inputs, targets)

        ## dice loss
        my_dice_loss = self.dice_loss(inputs, targets)
        ## cross entropy
        my_cross_entorpy = self.cross_entorpy(inputs, targets)

        #final_loss = 0.3 * my_iou_loss + 0.4 * my_dice_loss
        ###  iou*0.2  dice*0.4  cross entropy * 0.4
        final_loss = 0.2*my_iou_loss + 0.4*my_dice_loss + 0.4*my_cross_entorpy
        return final_loss





if __name__ == '__main__':
    cuda0 = torch.device('cuda:0')
    y_pred = torch.rand((8, 6, 512, 512), device=cuda0)
    y_true = torch.rand((8, 1, 512, 512), device=cuda0).long().squeeze(1)

    criterion = DiceCE()
    criterion.to(cuda0)
    loss = criterion(y_pred, y_true)
    print(loss)
