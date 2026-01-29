import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    """Computes the Dice Loss across multiple classes to handle regional overlap and class imbalance."""
    
    def __init__(self, smooth=1e-6, ignore_index=0):
        """Initializes loss with a smoothing factor to prevent division by zero and a target index to ignore."""
        super(MulticlassDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        Calculates Dice Loss by evaluating class-wise overlap between predictions and ground truth.
        
        Args:
            output (torch.Tensor): Raw model logits [Batch, Classes, H, W]
            target (torch.Tensor): Ground truth class labels [Batch, H, W]
        Returns:
            torch.Tensor: Scalar loss value (1 - Mean Dice Score)
        """
        num_classes = output.shape[1]
        # Convert logits to probabilities via Softmax across the channel dimension
        probs = F.softmax(output, dim=1)

        # Convert target to one-hot: [Batch, H, W] -> [Batch, Classes, H, W]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        dice_per_class = []

        # Iterate through classes (skip background if ignore_index is set)
        start_idx = 1 if self.ignore_index == 0 else 0

        for i in range(start_idx, num_classes):
            if i == self.ignore_index:
                continue

            # Flatten spatial dimensions to compute global intersection/union for the batch
            input_flat = probs[:, i, :, :].reshape(-1)
            target_flat = target_one_hot[:, i, :, :].reshape(-1)

            intersection = (input_flat * target_flat).sum()
            unionset = input_flat.sum() + target_flat.sum()

            # Apply Dice formula with smoothing to stabilize gradient calculation
            dice = (2. * intersection + self.smooth) / (unionset + self.smooth)
            dice_per_class.append(dice)

        # Average the Dice across all evaluated classes
        overall_dice = torch.stack(dice_per_class).mean()

        return 1 - overall_dice

class FocalLoss2d(nn.Module):
    """Implements Focal Loss to prioritize 'hard' pixels by down-weighting easy-to-classify examples."""
    def __init__(self, gamma=2.0, weight=None):
        """Initializes the focusing parameter (gamma) and optional class-wise weights."""
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight # Optional: Tensor of shape [Num_Classes]

    def forward(self, input, target):
        """
        Scales standard Cross Entropy loss based on the model's confidence in each pixel.
        
        Args:
            input (torch.Tensor): Raw model logits [Batch, Classes, H, W]
            target (torch.Tensor): Ground truth class indices [Batch, H, W]
        """
        # Calculate standard Cross Entropy (pixel-wise)
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')

        # Calculate probabilities
        pt = torch.exp(-ce_loss)

        # Calculate Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Return average
        return focal_loss.mean()

class LossMixer(nn.Module):
    """Hybrids two loss functions using a weighted sum to leverage multiple optimization objectives."""
  def __init__(self, loss_fn1, loss_fn2, alpha=0.5):
    """Sets the component loss functions and the alpha ratio for the weighted combination."""
    super().__init__()
    self.loss_fn1 = loss_fn1
    self.loss_fn2 = loss_fn2
    self.alpha = alpha

  def forward(self, input, target):
    """Computes the final loss as a linear combination of two distinct criteria."""
    loss1 = self.loss_fn1(input, target)
    loss2 = self.loss_fn2(input, target)

    # Weighted summation based on the alpha parameter
    total_loss = (self.alpha * loss1) + ((1 - self.alpha) * loss2)

    return total_loss
