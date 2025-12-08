from torch.utils.data import Dataset
import torch

class SimpleDataset(Dataset):
    def __init__(self, config):
        self.config = config

        max_word_length = config["max_word_length"]
        max_coord_length = config["max_coord_length"]

        self.words = torch.empty((0, max_word_length), dtype=int)
        self.targets = torch.empty((0, max_coord_length), dtype=int)

    def __len__(self):
        return len(self.state_cat)

    def __getitem__(self, index):
        sample = (
            self.state_cat[index],
            self.state_cont[index],
            self.targets[index]
        )

        return sample
    
    def save(self, location):
        torch.save(self, location)

    def append(self, states, verbose=False, **kwargs):
        state_cat, state_cont, targets = self.process_data(states, verbose=verbose)

        self.raw_append(state_cat, state_cont, targets)
    
    # assumes the data has already been processed
    def raw_append(self, state_cat, state_cont, targets):
        new_state_cat = torch.tensor(state_cat, dtype=int)
        new_state_cont = torch.tensor(state_cont, dtype=torch.float32)
        new_targets = torch.tensor(targets, dtype=int).reshape((-1,1))

        # probably overkill but ah well
        new_state_cat._fix_weakref()
        new_state_cont._fix_weakref()
        new_targets._fix_weakref()

        self.state_cat = torch.cat((self.state_cat, new_state_cat))
        self.state_cont = torch.cat((self.state_cont, new_state_cont))
        self.targets = torch.cat((self.targets, new_targets))

        # probably overkill but ah well
        self.state_cat._fix_weakref()
        self.state_cont._fix_weakref()
        self.targets._fix_weakref()

class Dynnikov(SimpleDataset):
    def process_data(self):
        return []


DATASETS = {
    "simple": SimpleDataset,
    "dynnikov": Dynnikov
}
