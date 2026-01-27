import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=0):
        super(MulticlassDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output: [Batch, Classes, H, W] (Raw logits)
        target: [Batch, H, W] (Class labels)
        """
        num_classes = output.shape[1]
        probs = F.softmax(output, dim=1)

        # Convert target to one-hot: [Batch, H, W] -> [Batch, Classes, H, W]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        dice_per_class = []

        # Iterate through classes (skip background if ignore_index is set)
        start_idx = 1 if self.ignore_index == 0 else 0

        for i in range(start_idx, num_classes):
            if i == self.ignore_index:
                continue

            input_flat = probs[:, i, :, :].reshape(-1)
            target_flat = target_one_hot[:, i, :, :].reshape(-1)

            intersection = (input_flat * target_flat).sum()
            unionset = input_flat.sum() + target_flat.sum()

            dice = (2. * intersection + self.smooth) / (unionset + self.smooth)
            dice_per_class.append(dice)

        # Average the Dice across all evaluated classes
        overall_dice = torch.stack(dice_per_class).mean()

        return 1 - overall_dice

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight # Optional: Tensor of shape [Num_Classes]

    def forward(self, input, target):
        """
        input: [Batch, Classes, H, W] - Raw logits (no Softmax yet)
        target: [Batch, H, W] - Class Indices (Long)
        """
        # 1. Calculate standard Cross Entropy (pixel-wise)
        # CRITICAL: reduction='none' preserves the loss for every pixel
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')

        # 2. Calculate probabilities (pt)
        pt = torch.exp(-ce_loss)

        # 3. Calculate Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Return average
        return focal_loss.mean()

class LossMixer(nn.Module):
  def __init__(self, loss_fn1, loss_fn2, alpha=0.5):
    super().__init__()
    self.loss_fn1 = loss_fn1
    self.loss_fn2 = loss_fn2
    self.alpha = alpha

  def forward(self, input, target):
    loss1 = self.loss_fn1(input, target)
    loss2 = self.loss_fn2(input, target)

    total_loss = (self.alpha * loss1) + ((1 - self.alpha) * loss2)

    return total_loss
