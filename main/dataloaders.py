import os, re
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, BatchSampler, DataLoader,Subset, SubsetRandomSampler
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as skl_LabelEncoder

from main import utils


class LitDataset(LightningDataModule):
    def __init__(self, cfg):
        super(LitDataset, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.exp.batch_size
        
        self.desired_composers = cfg.exp.composers
        train_dataset = globals()[cfg.exp.dataset]
        test_dataset =  globals()[cfg.exp.test_dataset if cfg.exp.test_dataset else cfg.exp.dataset]        
        
        # MAESTRO has |train|valid|test|, ASAP & ATEPP have |train|test|, SMD has |test| set only
        self.train_data = None if cfg.exp.testing_only else train_dataset(cfg, split='train')
        self.valid_data = None if cfg.exp.testing_only else train_dataset(cfg, split='validation')
        self.test_data = test_dataset(cfg, split='test')

        # Composer selection
        self.composer_classname = "canonical_composer" if cfg.exp.dataset != "ASAP" else "composer"
        if self.desired_composers == "All":
                pass  # Use all composers
        elif self.desired_composers == "Common":
            self._select_common(cfg.exp.uncommon_threshold)
        else:
            self._select_composers()

        # Custom validation set
        if not cfg.exp.testing_only and cfg.exp.dataset != "MAESTRO" and cfg.exp.custom_valid_ratio > 0:
            self.train_data, self.valid_data = self._train_valid_split(cfg.exp.custom_valid_ratio)

        # Labels
        self.train_labels = self.train_data.label_column.to_numpy() if self.train_data else None
        self.valid_labels = self.valid_data.label_column.to_numpy() if self.valid_data else None
        self.test_labels = self.test_data.label_column.to_numpy() if hasattr(self.test_data, 'label_column') else np.full(len(self.test_data), -1)

    def _select_common(self, threshold):
        """Drop the uncommon composers by provided threshold."""
        self.train_data.metadata = utils.drop_uncommon_classes(self.train_data.metadata, self.composer_classname, threshold) if self.train_data else None
        self.valid_data.metadata = utils.drop_uncommon_classes(self.valid_data.metadata, self.composer_classname, threshold) if self.valid_data else None
        # self.test_data.metadata = utils.drop_uncommon_classes(self.test_data.metadata, self.composer_classname, threshold)

    def _select_composers(self):
        """Select certain composers based on partial matching."""
        def match_composer(composer_name):
            """Match composer using partial name matching."""
            return any(re.search(r'\b' + re.escape(desired) + r'\b', composer_name, re.IGNORECASE) for desired in self.desired_composers)
        if self.train_data is not None:
            self.train_data.metadata = self.train_data.metadata[self.train_data.metadata[self.composer_classname].apply(match_composer)]
        if self.valid_data is not None:
            self.valid_data.metadata = self.valid_data.metadata[self.valid_data.metadata[self.composer_classname].apply(match_composer)]
        # if self.test_data is not None:
        #     self.test_data.metadata = self.test_data.metadata[self.test_data.metadata[self.composer_classname].apply(match_composer)]

    def _train_valid_split(self, valid_ratio):
        """Create a custom train|valid split from the training data."""
        orign_indices = list(range(len(self.train_data)))
        orign_labels = self.train_data.label_column.to_numpy()
        self.train_indices, self.valid_indices = train_test_split(orign_indices, test_size=valid_ratio, stratify=orign_labels, random_state=self.cfg.exp.random_seed)
        return Subset(self.train_data, self.train_indices), Subset(self.train_data, self.valid_indices)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.cfg.exp.num_workers) if self.train_data else None

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.cfg.exp.num_workers) if self.valid_data else None

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=self.cfg.exp.num_workers, shuffle=False)

# ----------------------------- NOTE: Self-defined Sampler ------------------------------------

class LengthSampler(BatchSampler):
    """Bucket the data that's of similar length into one batch, to avoid padding redundancy
    How it works: using the duration from metadata, order the pieces from short to long and then sample
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.metadata = dataset.dataset.metadata.iloc[dataset.indices]
        self.num_samples = len(dataset)
        self.ordered_index = self.order_by_length()
        
    def __iter__(self):
        for i in self.ordered_index:
            yield self.dataset.indices.index(i)
    
    def __len__(self):
        return self.num_samples
    
    def order_by_length(self):
        return self.metadata.sort_values(by=['duration'], ascending=False).index

# --------------------- NOTE: Setup for Specific Dataset -------------------------

class ASAP(Dataset):
    """Returns the (symbolic data, label) datapair from the ASAP dataset."""
    def __init__(self, cfg, split='train'):
        self.dataset_dir = cfg.dataset.ASAP.dataset_dir
        self.metadata = pd.read_csv(cfg.dataset.ASAP.metadata_file)
        self.metadata = self.metadata[self.metadata['composer_split_mid'] == split]
        
        self.label_column = self.metadata['composer']
        self.label_encoder = skl_LabelEncoder()
        self.label_encoder.fit(self.label_column.unique())

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.label_column.iloc[idx]
        label = self.label_encoder.transform([label])[0]
        return (self.dataset_dir + self.metadata['midi_performance'].iloc[idx], label)


class MAESTRO(Dataset):
    """Returns the (symbolic data, label) datapair from the MAESTRO dataset."""
    def __init__(self, cfg, split='train'):
        self.dataset_dir = cfg.dataset.MAESTRO.dataset_dir
        self.metadata = pd.read_csv(cfg.dataset.MAESTRO.metadata_file)
        self.metadata = self.metadata[self.metadata['split'] == split]

        self.label_column = self.metadata['canonical_composer']
        self.label_encoder = skl_LabelEncoder()
        self.label_encoder.fit(self.label_column.unique())

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.label_column.iloc[idx]
        label = self.label_encoder.transform([label])[0]
        return (self.dataset_dir + self.metadata['midi_filename'].iloc[idx], label)


class ATEPP(Dataset):
    """Returns the (symbolic data, label) datapair from the ATEPP dataset."""
    def __init__(self, cfg, split='train'):
        self.dataset_dir = cfg.dataset.ATEPP.dataset_dir
        self.metadata = pd.read_csv(cfg.dataset.ATEPP.metadata_file, encoding="iso-8859-1")
        self.scp_path = utils.read_scp_file('/home/zhe/vae/conf/directories.scp')
        # self.metadata = pd.read_csv(cfg.dataset.ATEPP.metadata_file)
        # self.metadata = drop_uncommon_classes(self.metadata, 'composer')
        self.metadata = self.metadata[self.metadata['composer_split_mid'] == split]
        
        self.label_column = self.metadata['composer']
        self.label_encoder = skl_LabelEncoder()
        self.label_encoder.fit(self.label_column.unique())

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.label_column.iloc[idx]
        label = self.label_encoder.transform([label])[0]
        mid_file_name = os.path.basename(self.metadata['midi_path'].iloc[idx])
        
        return (utils.find_mid_file_in_list(self.scp_path, mid_file_name), label)


class SMD(Dataset):
    """Returns the (symbolic data, -1) datapair from the SMD dataset, -1 is placeholder meaning no label."""
    def __init__(self, cfg, split='test'):
        self.dataset_dir = cfg.dataset.SMD.dataset_dir
        if split != 'test':
            raise ValueError("SMD dataset only supports the 'test'.")
        
        # Filter out "Beethoven_WoO080" because it wasn't used in previous study
        self.midi_files = [
            f for f in os.listdir(self.dataset_dir)
            if (f.endswith('.midi') or f.endswith('.mid')) and f != "Beethoven_WoO080_001_20081107-SMD.mid"
        ]

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_file = os.path.join(self.dataset_dir, self.midi_files[idx])
        return midi_file, -1