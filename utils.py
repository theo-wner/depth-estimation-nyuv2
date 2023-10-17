import os
import matplotlib.pyplot as plt

"""
Defines utility functions
"""

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