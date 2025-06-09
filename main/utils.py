import os, random, gc, json, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback

# ----------------------- NOTE Checkpoint Operation in train.py ------------------------------------------

def find_checkpoints(checkpoint_dir: str) -> list:
    """Find all checkpoint files and return sorted epoch numbers."""
    return sorted([m.group(1) for m in [re.search(r'epoch=(\d+)\.ckpt$', f) 
                  for f in os.listdir(checkpoint_dir)] if m], key=int)


# ----------------------- NOTE Weights Initialization ---------------------------


def init_xavier(m):
    """Applies Xavier uniform initialization to linear and convolutional layers."""
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_uniform(m):
    """Applies uniform initialization to linear and convolutional layers."""
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.uniform_(m.weight, -0.1, 0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_kaiming(m):
    """Applies Kaiming (He) initialization to linear and convolutional layers."""
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_no_op(m):
    """A no-operation function, does not alter weights."""
    pass

def initialize_weights(model, init_type='no_init', component='Linear'):
    """Initializes weights of a model based on the specified type and component."""
    init_func = {
        'xavier': init_xavier,
        'uniform': init_uniform,
        'kaiming': init_kaiming,
        'no_init': init_no_op     # Explicitly handle no initialization case
    }.get(init_type, init_no_op)  # Default to no-op if init_type is unknown or 'no_init'

    if component == 'Linear':
        for m in model.modules():
            if isinstance(m, nn.Linear):
                init_func(m)
    elif component == 'GRU' or component == 'LSTM':
        for m in model.modules():
            if isinstance(m, (nn.GRU, nn.LSTM)):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:    # Input-to-hidden weights
                        init_func(param)
                    elif 'weight_hh' in name:  # Hidden-to-hidden weights
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:       # Bias initialization
                        nn.init.constant_(param, 0)
    elif component == 'Conv':
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init_func(m)
    else:
        model.apply(init_func)  # Apply to all layers


# ----------------------- NOTE Dataloading Functions ---------------------------

def drop_uncommon_classes(metadata, label_col, threshold=0.03): # threshold=0.03
    """sort the classes by distribution and drop the data that consists of the last threshold\% of the set"""
    count = metadata[label_col].value_counts().to_frame("count")
    count['agg_percent'] = count.loc[::-1, 'count'].cumsum()[::-1] / count.sum().values
    uncommon_label = count[count['agg_percent'] < threshold].index
    return metadata[~metadata[label_col].isin(uncommon_label)]


def find_mid_file_in_list(paths, mid_file_name):
    for path in paths:
        if mid_file_name in path:
            return path
    return None


def read_scp_file(scp_file):
    with open(scp_file, 'r') as file:
        paths = [line.strip() for line in file.readlines()]
    return paths

# -------------------- NOTE Training Preparation Functions ---------------------------

def load_data(path, cfg):
    """generic load data function for any type of representation """
    save_dir = f"{cfg.matrix.save_dir}/{cfg.matrix.save_folder}"

    if not os.path.exists(save_dir):
        return None

    metadata = pd.read_csv(f"{save_dir}/metadata.csv")
    res = metadata[metadata['path'] == path]
    if len(res):
        return np.load(f"{save_dir}/{res['save_dir'].iloc[0]}")


def save_data(path, computed_data, cfg):
    """generic save_data function for any type of representation
    - write the corresponding path with the saved index in metadata.csv
    graphs: dgl 
    matrix and sequence: numpy npy
    """
    save_dir = f"{cfg.matrix.save_dir}/{cfg.matrix.save_folder}"

    if not os.path.exists(save_dir): # make saving dir if not exist
        os.makedirs(save_dir)
        with open(f"{save_dir}/metadata.csv", "w") as f:
            f.write("path,save_dir\n")

    metadata = pd.read_csv(f"{save_dir}/metadata.csv")
    if path in metadata['path']: # don't write and save if it existed
        return

    N = len(metadata) 

    save_path = f"{N}.npy"
    np.save(f"{save_dir}/{save_path}", computed_data)
    
    # metadata = metadata.append({"path": path, "save_dir": save_path}, ignore_index=True)
    new_row = pd.DataFrame({"path": [path], "save_dir": [save_path]})
    metadata = pd.concat([metadata, new_row], ignore_index=True)
    metadata.to_csv(f"{save_dir}/metadata.csv", index=False)


def pad_batch(b, cfg, device, batch_data, batch_labels):
    """padding batch: 
    1. refill value to batch size: when the processed batch lost data because of parsing error, 
        refill the batch with the last one in the batch
    2. for batch with variable segments length, pad the shorter data util they have the same
        number of segments.
        - For matrix: also pad with all-zero matrices
        - For sequence: pad the remaining segments with 0 (a non-vocab value)
    """
    # refill
    if not batch_data:
        batch_data = [np.zeros((1, cfg.sequence.max_seq_len, 6))]
        batch_labels = [0]
    n_skipped = b - len(batch_data)
    batch_data += [batch_data[-1]] * n_skipped
    batch_labels = torch.tensor(batch_labels + [batch_labels[-1]] * n_skipped, device=device)

    # pad seg
    max_n_segs = max([len(data) for data in batch_data])
    batch_data = [*map(lambda data: np.concatenate((data, np.zeros((max_n_segs - len(data), *data.shape[1:])))), batch_data)]
    batch_data = [np.concatenate((data, np.zeros((max_n_segs - len(data), *data.shape[1:])))) for data in batch_data]
    return batch_data, batch_labels


def prep_run(cfg):
    """
    prepare gpu memory, random seed, empty cache, clean wandb logs
    """
    gc.collect()
    torch.manual_seed(cfg.exp.random_seed)
    random.seed(cfg.exp.random_seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.empty_cache()
    os.system("wandb sync --clean-force --clean-old-hours 3")
    return


# ----------------- NOTE Self-defined Callback & LRScheduler ---------------------------


class TestEveryNEpochs(Callback):
    """
    run the test every `every_n_epochs`
    """
    def __init__(self, test_dataloader, every_n_epochs):
        self.test_dataloader = test_dataloader
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:

            pl_module.eval()  # Set model to evaluation mode
            test_steps = 0
            for batch in self.test_dataloader:
                with torch.no_grad():  # Use test_step for evaluation
                    mse = pl_module.test_step(batch, batch_idx=None)
                    test_steps += 1
            # print(f"Test MSE after epoch {trainer.current_epoch + 1}: {mse}")
            pl_module.train()  # Set the model back to training mode


class WarmupLinearLRSchedule:
    """
    Implements Warmup and Warmdown learning rate scheduler. Using 'warmup_epochs' going from 'init_lr' to 'lr', then automatically warmdown from "lr" to "end_lr"
    from https://github.com/dome272/MaskGIT-pytorch/blob/cff485ad3a14b6ed5f3aa966e045ea2bc8c68ad8/lr_schedule.py#L4
    """
    def __init__(self, optimizer, init_lr, peak_lr, end_lr, warmup_epochs, epochs=100, current_step=0):
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.optimizer = optimizer
        self.warmup_rate = (peak_lr - init_lr) / warmup_epochs
        self.decay_rate = (end_lr - peak_lr) / (epochs - warmup_epochs)
        self.update_steps = current_step
        self.lr = init_lr
        self.warmup_steps = warmup_epochs
        self.epochs = epochs
        if current_step > 0:
            self.lr = self.peak_lr + self.decay_rate * (current_step - 1 - warmup_epochs)

    def set_lr(self, lr):
        print(f"Setting lr: {lr}")
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def step(self):
        if self.update_steps <= self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
        # elif self.warmup_steps < self.update_steps <= self.epochs:
        else:
            lr = max(0., self.lr + self.decay_rate)
        self.set_lr(lr)
        self.lr = lr
        self.update_steps += 1
        return self.lr
    
# ------------- NOTE: Unuseful ------------------------
'''
def get_pc_one_hot(note_array):
    """Get one-hot encoding of pitch class."""
    one_hot = np.zeros((len(note_array), 12))
    idx = (np.arange(len(note_array)),np.remainder(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot


def get_octave_one_hot(note_array):
    """Get one-hot encoding of octave."""
    one_hot = np.zeros((len(note_array), 10))
    idx = (np.arange(len(note_array)), np.floor_divide(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot


def get_onset_one_hot(note_events):
    """Get one-hot encoding of onset within the 60s segment"""
    seg_time = 60
    one_hot = np.zeros((len(note_events), seg_time))
    onsets = np.array(note_events["start"]) % seg_time
    idx = (np.arange(len(note_events)),np.remainder(note_events["pitch"], 12))
    one_hot[idx] = 1
    return one_hot


def get_pedal_one_hot(note_events):
    """Get one-hot encoding of sustain pedal values."""
    one_hot = np.zeros((len(note_events), 8))
    idx = (np.arange(len(note_events)), np.floor_divide(note_events["sustain_value"], 16).astype(int))
    one_hot[idx] = 1
    return one_hot


def get_velocity_one_hot(note_events):
    """Get one-hot encoding of velocity values."""
    one_hot = np.zeros((len(note_events), 8))
    idx = (np.arange(len(note_events)), np.floor_divide(note_events["velocity"], 16).astype(int))
    one_hot[idx] = 1
    return one_hot
'''