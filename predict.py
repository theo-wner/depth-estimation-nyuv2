import torch
from model import DepthFormer
from dataset import NYUv2Dataset
from tqdm import tqdm
import transformers

from utils import visualize_img_depth
import config

"""
Predicts with the model
"""

if __name__ == '__main__':
    # Set the verbosity of the transformers library to error
    transformers.logging.set_verbosity_error()

    # Initialize the model (and load it to the CPU)
    model = DepthFormer()

    if config.CPU_USAGE:
        checkpoint = torch.load(config.CHECKPOINT, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = DepthFormer.load_from_checkpoint(config.CHECKPOINT)
        model = model.to(config.DEVICES[0])
    model.eval()

    # Dataset
    dataset = NYUv2Dataset(split='test')

    # Predict
    for i in tqdm(range(50)):
        image, depth = dataset[i]

        if not config.CPU_USAGE:
            image = image.to(config.DEVICES[0])

        with torch.no_grad():
            pred = model(image.unsqueeze(0))[0]
            pred = torch.nn.functional.interpolate(pred, size=image.shape[-2:], mode="bilinear", align_corners=False)    # upsample logits to input image size (SegFormer outputs h/4 and w/4 by default, see paper)
            pred = pred.squeeze()

        # Mask out the predictions where the original image is black in case you predict augmented train images
        pred[image.sum(dim=0) == 0] = 255

        # Visualization
        image = image.cpu()
        depth = depth.cpu()
        pred = pred.cpu()
        visualize_img_depth(image, depth, pred, filename=f'test_{i}.png')

