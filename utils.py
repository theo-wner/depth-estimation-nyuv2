import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler

"""
Defines utility functions
"""

# LR = Initial_LR * (1 - iter / max_iter)^0.9
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iterations, power=0.9):
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        self.current_iteration += 1
        return [base_lr * (1 - self.current_iteration / self.max_iterations) ** self.power for base_lr in self.base_lrs]


# Creates a tuple of image and Ground Truth depth map
def visualize_img_depth(image, gt_depth, pr_depth, filename='test.png'):
    # Create a new figure
    plt.figure(figsize=(18, 5))

    # Place Subplots
    plt.subplots_adjust(left=0.005,
                        bottom=0,
                        right=0.95,
                        top=1,
                        wspace=0.25,
                        hspace=0.0)

    # Plot Image
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image.permute(1, 2, 0).numpy())

    # Plot Ground Truth Depth Map
    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('Spectral')
    norm = plt.Normalize(vmin=0, vmax=40)
    plt.imshow(gt_depth.squeeze().numpy(), cmap=cmap, norm=norm)

    # Plot Predicted Depth Map
    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    cmap = plt.cm.get_cmap('Spectral')
    norm = plt.Normalize(vmin=0, vmax=40)
    plt.imshow(pr_depth.squeeze().numpy(), cmap=cmap, norm=norm)

    # Add Colorbar to the right of the plots with original depth values
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), 
                        ax=plt.gcf().get_axes(), 
                        orientation='vertical', 
                        pad=0.02,
                        fraction=0.012)
    cbar.set_label('Depth')

    # Save Figure
    directory = './figures'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, filename))
    plt.close()