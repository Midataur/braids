from utilities import *
from torch.utils.data import DataLoader
from dataset_types import DATASETS
import numpy as np
import torch
import os

# subset can be "train", "val", or "test"
def create_dataset(subset, config, verbose=False):
    DataSetType = DATASETS[config["model_type"]]

    # sometimes this can be "verbose AND some other condition"
    should_speak = verbose

    # do train states
    if should_speak:
        print(f"Loading {subset} data...")
        
    path = f"./datasets/{config["dataset"]}/{subset}"
    
    # load data
    filenames = os.listdir(path)
    dataset = DataSetType(config)

    for filename in filenames:
        data = np.loadtxt(filename, delimiter=",", dtype=int)
        dataset.append(data)
        
    return dataset

# subset can be "train", "val", or "test"
def get_dataset_and_loader(subset, config, verbose=False):
    if verbose:
        print(f"Creating {subset} dataset...")
    
    dataset = create_dataset(subset, config, verbose)

    batchsize, n_workers = config["batchsize"], config["n_workers"]
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=n_workers)

    return dataset, dataloader