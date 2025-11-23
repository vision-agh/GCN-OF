import yaml
import lightning as L

from omegaconf import omegaconf
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import Callback

import h5py

data = h5py.File('data/indoor_flying1_data.hdf5')
print(data['davis']['left']['image_raw'])