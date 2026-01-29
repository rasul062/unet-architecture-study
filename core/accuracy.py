import torch.nn as nn

class MulticlassIOU(nn.Module):
    """Computes Intersection over Union (IoU) for multiclass semantic segmentation."""
    def __init__(self, num_classes):
        """Initializes the metric with the total number of dataset classes."""
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred_mask, target_mask):
        """
        Calculates IoU by comparing predicted class indices against ground truth.
        
        Args:
            pred_mask (torch.Tensor): Predicted class indices [Batch, H, W]
            target_mask (torch.Tensor): Ground truth class indices [Batch, H, W]
        Returns:
            torch.Tensor: Scalar IoU value for the first detected class in the target.
        """

        # Loop over every class
        for cls_id in range(self.num_classes):
            # Create binary masks for this specific class
            pred_inds = (pred_mask == cls_id)
            target_inds = (target_mask == cls_id)

            # If this class is not in the target, skip it
            if target_inds.long().sum().item() == 0:
                continue

            # Calculate Intersection and Union using Logical operators
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()

            # Avoid dividing by zero
            if union.item() != 0:
                return intersection / union
