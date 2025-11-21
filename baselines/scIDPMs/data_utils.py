import anndata
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import argparse
import h5py
from typing import Dict, Any, List

class SingleCellDatasetBaselines(Dataset):
    """
    Custom PyTorch Dataset class for scRNA data.
    """
    def __init__(self, data_source: str, 
                 split: str = 'train', 
                 condition_keys: list[str] = None, 
                 missing_ratio: float = 0.1, 
                 imputation_mask: str = None,
                 seed: int = 42):
        """
        Args:
            data_source (str): Path to the .h5 or .hdf5 file containing preprocesed data.
            split (str): which data split for the dataset (train/val/test).
            condition_keys (list[str], optional): List of condition keys to load from .obs (e.g. ["cell_type, "batch"]).
            return_strings_for (list[str], optional): List of condition keys to return string instead of category code. 
            """
        super().__init__()
        print(f"\n--- Initializing Dataset for {split} split ---")
        self.split = split
        self.condition_keys = (
            condition_keys if condition_keys is not None else ["cell_type", "batch"]
        )
        self.gene_names = None
        self.seed = seed
        self.missing = missing_ratio
        if isinstance(data_source, str) and data_source.endswith((".h5", ".hdf5")):
            self._init_from_h5(data_source)
        else:
            raise TypeError("data_source must be a path to a pre-processed .h5 file.")

        self.observed_mask = ~self.missingness_mask.numpy().copy()
        self.true_mask = self.missingness_mask.numpy().copy()

        if imputation_mask is not None:
            print("Using provided imputation mask.")
            if not isinstance(imputation_mask, str) and imputation_mask.endswith((".npy")):
                raise TypeError("imputation_mask should be a path to a defined mask")
            loaded_mask = np.load(imputation_mask).astype(bool)
            if loaded_mask.shape != self.counts.shape:
                raise ValueError(
                    f"Provided imputation_mask shape {loaded_mask.shape} "
                    f"does not match counts shape {self.counts.shape}"
                )
            target_mask = loaded_mask * self.observed_mask.copy()
            self.gt_mask = self.observed_mask.copy() * ~target_mask
        else:
            print(f"Generating new random imputation mask with missing ratio: {missing_ratio}")
            self._generate_random_mask(missing_ratio, seed)
        print("Dataset initialized.")
        print(f"Number of cells: {len(self)}")
        print(f"Number of genes: {self.counts.shape[1]}")

    def _init_from_h5(self, file_path):
        """
        Initializes Dataset object from hdf5 path.
        """
        print(f"Loading from HDF5 file: '{file_path}'...")
        with h5py.File(file_path, "r") as f:
            if self.split not in f:
                raise ValueError(f"Split '{self.split}' not found in HDF5 file")
            split_group = {k.lower(): v for k, v in f[self.split].items()}
            self.counts = torch.tensor(split_group["counts"][:], dtype=torch.float32)
            self.missingness_mask = torch.tensor(
                split_group["missingness_mask"][:], dtype=torch.bool
            )
            self.gene_names = [name.decode('utf-8') for name in f['gene_names'][:]]

            for key in self.condition_keys:
                key = key.lower()
                value_key = f"{key}_values"
                if value_key not in split_group:
                    print(f"Warning: Key '{value_key}' not in split. Skipping.")
                    continue
                string_values = [s.decode('utf-8') for s in split_group[value_key][:]]
                cats = pd.Categorical(string_values)
                setattr(
                    self, f"{key}_labels", torch.tensor(cats.codes, dtype=torch.long)
                )
                setattr(self, f"{key}_mapping", dict(enumerate(cats.categories)))

                

    def __len__(self):
        return self.counts.shape[0]

    # def _generate_masks(self):
    #     """
    #     generates masks for dataset
    #     true_masks: marks all biological zeros (original missingness)
    #     observed_masks: marks all non-zeros
    #     gt_mask: marks values given to model (subset of observed_mask)
    #     """
    #     self.true_mask = (self.missingness_mask).numpy()
    #     self.observed_mask = ~self.missingness_mask

    #     np.random.seed(self.seed)
    #     gt_mask = self.observed_mask.numpy().copy()

    #     for col_idx in range(self.counts.shape[1]):
    #         obs_indices = np.where(self.observed_mask[:, col_idx] == 1)[0]
    #         if len(obs_indices) > 0:
    #             # Choose a fraction of these to hide
    #             num_to_hide = int(len(obs_indices) * self.missing)
    #             hide_indices = np.random.choice(obs_indices, size = num_to_hide, replace = False)

    #             gt_mask[hide_indices, col_idx] = 0
    #     self.gt_mask = gt_mask

    def _generate_random_mask(self, dropout, seed = 42):
        """
        Generates random dropout masking for imputation
        """
        np.random.seed(seed)
        gt_mask = self.observed_mask.copy()
        for col_idx in range(self.counts.shape[1]):
            obs_indices = np.where(self.observed_mask[:, col_idx])[0]
            
            if len(obs_indices) > 0:
                num_to_hide = int(len(obs_indices) * dropout)
                if num_to_hide > 0:
                    hide_indices = np.random.choice(
                        obs_indices, size=num_to_hide, replace=False
                    )
                    gt_mask[hide_indices, col_idx] = False
                    
        self.gt_mask = gt_mask

    def __getitem__(self, idx):
        items = []
        for key in self.condition_keys:
            code = getattr(self, f"{key}_labels")[idx].item()
            items.append(torch.tensor(code, dtype=torch.long))

        
        s = {
            "observed_data": self.counts[idx].clone().detach(),
            "observed_mask": self.observed_mask[idx],
            "gt_mask": torch.tensor(self.gt_mask[idx], dtype = torch.bool),
            "timepoints": torch.arange(self.counts.shape[1], dtype=torch.long),
            "true_masks": torch.tensor(self.true_mask[idx], dtype = torch.bool)
        }
        return s
        
    def get_num_classes(self, key):
        """
        Returns the number of unique classes for a given condition key.
        """
        if key not in self.condition_keys:
            raise ValueError(f"Key '{key}' not in condition_keys: {self.condition_keys}")
        mapping = getattr(self, f"{key}_mapping", None)
        if mapping is None:
            raise ValueError(f"No mapping found for key '{key}'")
        return len(mapping)
    def get_conditions_df(self) -> pd.DataFrame:
        """
        Constructs a pandas DataFrame with all condition key values for the entire dataset.

        Returns:
            pd.DataFrame: A DataFrame where each column corresponds to a condition_key
                          and each row corresponds to a cell.
        """
        conditions_data = {}
        for key in self.condition_keys:
            key = key.lower()
            # Get the tensor of integer codes and the code-to-string mapping
            label_codes = getattr(self, f"{key}_labels").numpy()
            mapping = getattr(self, f"{key}_mapping")
            
            # Map the integer codes back to their original string values
            string_labels = [mapping[code] for code in label_codes]
            
            # Add the list of string labels as a new column
            conditions_data[key] = string_labels
            
        return pd.DataFrame(conditions_data)
    
    def get_obs_dict(self, unique: bool = False) -> Dict[str, List[Any]]:

        """

        Generates a dictionary mapping each column name to a list of its unique values.

        """
        if unique:
            return {col: self.get_conditions_df()[col].unique().tolist() for col in self.get_conditions_df().columns}
        else:
            return {col: self.get_conditions_df()[col].tolist() for col in self.get_conditions_df().columns}
