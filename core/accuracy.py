import torch.nn as nn

class MulticlassIOU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred_mask, target_mask):
        """
        pred_logits: [Batch, 21, H, W] (Output of the model)
        target_mask: [Batch, H, W]     (Ground Truth)
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
