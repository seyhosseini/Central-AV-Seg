import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        num_classes = output.size(1)
        loss = 0.0

        for class_idx in range(num_classes):
            output_class = output[:, class_idx, ...]
            target_class = target[:, class_idx, ...]

            intersection = torch.sum(output_class * target_class)
            union = torch.sum(output_class) + torch.sum(target_class)
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

            loss -= torch.log(dice_score)

        return loss / num_classes