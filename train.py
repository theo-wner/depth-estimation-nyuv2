import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import config
from dataset import NYUv2DataModule
from model import SegFormer
import transformers

"""
Trains the model
"""

if __name__ == '__main__':
    # Set the verbosity of the transformers library to error
    transformers.logging.set_verbosity_error()

    # Initialize the logger
    logger = TensorBoardLogger('logs', name='segformer')

    # Initialize the data module
    data_module = NYUv2DataModule(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    # Initialize the model
    model = SegFormer()

    # Initialize the trainer
    if config.CPU_USAGE:
        trainer = pl.Trainer(logger=logger, max_epochs=config.NUM_EPOCHS, accelerator='cpu', precision=config.PRECISION)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=config.NUM_EPOCHS, accelerator='gpu', precision=config.PRECISION, devices=config.DEVICES)

    # Train the model
    trainer.fit(model, data_module, ckpt_path=config.CHECKPOINT)