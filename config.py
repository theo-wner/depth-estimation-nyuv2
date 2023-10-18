import argparse

"""
Defines the Hyperparameters as command line arguments
"""

parser = argparse.ArgumentParser(description='Parser')

parser.add_argument('--backbone', type=str, default='b5', help='Backbone of the model')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of Workers')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of Epochs')
parser.add_argument('--learning_rate', type=float, default=6e-5, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay')
parser.add_argument('--precision', type=int, default=16, help='Precision')
parser.add_argument('--devices', type=int, nargs='+', default=[1], help='Devices')
parser.add_argument('--cpu_usage', type=bool, default=False, help='CPU Usage')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint')

args = parser.parse_args()

# Dataset
NUM_CLASSES = 1 # For Depth Estimation
IGNORE_INDEX = 255
NUMBER_TRAIN_IMAGES = 795
NUMBER_VAL_IMAGES = 654

# Model
BACKBONE = args.backbone

# Training
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
PRECISION = args.precision
DEVICES = args.devices
CPU_USAGE = args.cpu_usage
CHECKPOINT = args.checkpoint