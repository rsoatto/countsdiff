import anndata
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import argparse
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List
import scipy.sparse as sp

class SingleCellProcessor:
    """
    Handles loading, filtering, and saving of single-cell data.
    """

    def __init__(self, file_path, n_top_genes=1000, min_cells=100000, val_size = 0.15, test_size = 0.15, is_count = True, gene_list_path = None):
        self.file_path = file_path
        self.n_top_genes = n_top_genes
        self.min_cells = min_cells
        self.adata_raw = None
        self.adata_processed = None
        self.val_size = val_size
        self.test_size = test_size
        self.split_indices = {}
        self.is_count = is_count
        self.gene_list = None
        if gene_list_path is not None:
            gene_list = []
            with open(gene_list_path, "r") as f:
                for line in f:
                    gene_list.append(line.strip())
            self.gene_list = gene_list

    def _get_highly_variable_genes(self):
        """
        Identifies most highly variable genes by coefficient of variation.
        """

        print(f"Finding the top {self.n_top_genes} most variable genes...")
        n_cells_per_gene = self.adata_raw.X.getnnz(axis=0)
        expressed_gene_indices = np.where(n_cells_per_gene >= self.min_cells)[0]

        if len(expressed_gene_indices) < self.n_top_genes:
            print(
                f"Warning: Only {len(expressed_gene_indices)} genes were found in >= {self.min_cells} cells."
            )
            return self.adata_raw.var_names[expressed_gene_indices]

        adata_expressed = self.adata_raw[:, expressed_gene_indices]
        counts_matrix = adata_expressed.X.tocsr()
        mean = counts_matrix.mean(axis=0).A1
        counts_squared = counts_matrix.power(2)
        mean_sq = counts_squared.mean(axis=0).A1
        var = mean_sq - np.power(mean, 2)
        std = np.sqrt(var + 1e-12)
        cv = std / (mean + 1e-8)
        top_subset_indices = np.argsort(cv)[-self.n_top_genes :]
        highly_variable_genes = adata_expressed.var_names[top_subset_indices]

        print(
            f"Found {len(highly_variable_genes)} highly variable genes after filtering."
        )
        return highly_variable_genes

    def _generate_missingness_mask(self):
        """
        Generates a missingness mask based on zero entries with cell type grouping.
        """
        print("Generating missingness mask...")
        adata = self.adata_processed
        mask = np.zeros(adata.shape, dtype=bool)
        for cell_type in adata.obs["cell_type"].unique():
            row_indices = np.where(adata.obs["cell_type"] == cell_type)[0]
            sub_matrix = adata.X[row_indices, :]
            for j in range(sub_matrix.shape[1]):
                col_data = sub_matrix[:, j]
                if col_data.nnz > 0:
                    zero_indices_in_group = np.where(col_data.toarray() == 0)[0]
                    original_row_indices_to_mask = row_indices[zero_indices_in_group]
                    mask[original_row_indices_to_mask, j] = True
        self.adata_processed.obsm["missingness_mask"] = mask
        print("Missingness mask generated.")

    def _split_data(self):
        """
        splits data into train/val/test splits
        """
        print("\n -- Splitting Data ---")
        adata = self.adata_processed
        cell_indices = np.arange(adata.n_obs)

        train_val_idx, test_idx = train_test_split(
            cell_indices,
            test_size=self.test_size,
            random_state=42 # Ensures reproducibility
        )

        relative_val_size = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_size,
            random_state=42 # Ensures reproducibility
        )

        self.split_indices = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
        }
        print(f"Split complete:")
        print(f"  Train size: {len(train_idx)} cells")
        print(f"  Validation size: {len(val_idx)} cells")
        print(f"  Test size: {len(test_idx)} cells")

    def _subset_adata_from_gene_list(self):
        """
        Takes a list of genes, iterates through, and takes the correct column or generates a column of all 0s
        """
        print("Using provided gene list")
        new_var = pd.DataFrame(index = self.gene_list)

        if sp.issparse(self.adata_raw.X):
            new_X = sp.lil_matrix((self.adata_raw.n_obs, len(self.gene_list)), dtype = self.adata_raw.X.dtype)
        else:
            new_X = np.zeros((self.adata_raw.n_obs, len(self.gene_list)), dtype = self.adata_raw.X.dtype)

        present_genes = [gene for gene in self.gene_list if gene in self.adata_raw.var_names]
        source_idxs = self.adata_raw.var_names.get_indexer(present_genes)
        target_idxs = new_var.index.get_indexer(present_genes)
        new_X[:, target_idxs] = self.adata_raw.X[:, source_idxs]

        if sp.issparse(new_X):
            new_X = new_X.tocsr()

        new_adata = anndata.AnnData(
                X = new_X,
                obs = self.adata_raw.obs.copy(),
                var = new_var
        )
        return new_adata
    
    def process(self):
        """
        Main preprocessing function for scRNA dataset.
        """
        print("\n--- Loading Data ---")
        self.adata_raw = anndata.read_h5ad(self.file_path)
        print("\n--- Filtering Genes ---")
        if self.gene_list is None:
            hvg_names = self._get_highly_variable_genes()
            self.adata_processed = self.adata_raw[:, hvg_names].copy()
        else:
            self.adata_processed = self._subset_adata_from_gene_list()

        self._generate_missingness_mask()

        self._split_data()
        return self.adata_processed

    def save_to_h5(self, output_path, condition_keys=None):
        """
        Saves processed data to HDF5 file.
        """
        if self.adata_processed is None:
            raise RuntimeError(
                "Data has not been processed yet. Call .process() first."
            )
        if condition_keys is None:
            condition_keys = ["cell_type", "batch"]

        print(f"\n--- Saving processed data to '{output_path}' ---")

        hvg_names = self.adata_processed.var_names
        if self.is_count:
            raw_counts_filtered_genes = self.adata_processed[:, hvg_names].X
        else:
            raw_counts_filtered_genes = self.adata_processed.raw[:, hvg_names].X

        with h5py.File(output_path, "w") as f:
            gene_names_str = self.adata_processed.var_names.astype(str).values
            f.create_dataset('gene_names', data=gene_names_str, dtype=h5py.string_dtype(encoding='utf-8'))

            for split_name, indices in self.split_indices.items():
                print(f"  Saving '{split_name}' split...")
                split_group = f.create_group(split_name)
                
                split_counts = raw_counts_filtered_genes[indices, :].toarray()
                split_group.create_dataset("counts", data=split_counts)

                split_group.create_dataset(
                    "missingness_mask", data=self.adata_processed.obsm["missingness_mask"][indices, :]
                )
                for key in condition_keys:
                    if key not in self.adata_processed.obs.columns:
                        continue
                    string_data = self.adata_processed.obs[key].iloc[indices].astype(str).values
                    split_group.create_dataset(
                        f"{key}_values",
                        data=string_data,
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
        print("Save complete.")


class SingleCellDataset(Dataset):
    """
    Custom PyTorch Dataset class for scRNA data.
    """
    def __init__(self, data_source: str, split: str = 'train', condition_keys: list[str] = None, return_strings_for: list[str] = None):
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
        self.return_strings_for = (
            return_strings_for if return_strings_for is not None else []
        )

        self.gene_names = None

        if isinstance(data_source, str) and data_source.endswith((".h5", ".hdf5")):
            self._init_from_h5(data_source)
        else:
            raise TypeError("data_source must be a path to a pre-processed .h5 file.")

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
                setattr(
                    self, f"{key}_values", np.array(string_values)
                )
                setattr(self, f"{key}_mapping", dict(enumerate(cats.categories)))

    def __len__(self):
        return self.counts.shape[0]

    def __getitem__(self, idx):
        items = []
        for key in self.condition_keys:
            code = getattr(self, f"{key}_labels")[idx].item()
            if key in self.return_strings_for:
                string_val = getattr(self, f"{key}_mapping")[code]
                items.append(string_val)
            else:
                items.append(torch.tensor(code, dtype=torch.long))
        if hasattr(self, "target_mask"):
            return tuple(
                [self.counts[idx], items, self.missingness_mask[idx], self.target_mask[idx]] 
            )
        else:
            return tuple(
                [self.counts[idx], items, self.missingness_mask[idx]] 
            )
        
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
    
    def build_covariate_dict(self, labels):
        """
        Builds a dictionary of named covariates from a list of numerical labels
        """
        output = {}
        for i, key in enumerate(self.condition_keys):
            codes = labels[i]
            if isinstance(codes, torch.Tensor) or isinstance(codes, np.ndarray):
                codes = codes.tolist()
            mapping = getattr(self, f"{key}_mapping", None)
            if mapping is None:
                raise ValueError(f"No mapping found for key '{key}'")
            output[key] = [mapping[code] for code in codes]
        return output
    
    def build_covariate_df(self, labels):
        """
        Builds a pandas DataFrame of named covariates from a list of numerical labels
        """
        covariate_dict = self.build_covariate_dict(labels)
        return pd.DataFrame(covariate_dict)
    
    def get_obs_dict(self, unique: bool = False) -> Dict[str, List[Any]]:

        """

        Generates a dictionary mapping each column name to a list of its unique values.

        """
        if unique:
            return {col: self.get_conditions_df()[col].unique().tolist() for col in self.get_conditions_df().columns}
        else:
            return {col: self.get_conditions_df()[col].tolist() for col in self.get_conditions_df().columns}
        
    def add_masks(self, target_mask):
        """
        Adds target masks to the dataset object to make dataloader use easier in imputation
        """
        self.target_mask = target_mask

    
    


def main(args):
    """
    Main function to run the data processing and loading pipeline.
    
    Example Usage from Terminal:
    python process_scrna.py \
        --input_file path/to/your/real_data.h5ad \
        --output_file path/to/your/processed_data.h5 \
        --n_top_genes 1000 \
        --min_cells 700 \
        --condition_keys cell_type batch \
        --test_size 0.1 \
        --val_size 0.1
    """

    processor = SingleCellProcessor(
        file_path=args.input_file,
        n_top_genes=args.n_top_genes,
        min_cells=args.min_cells,
        val_size = args.val_size,
        test_size = args.test_size,
        is_count = args.is_count,
        gene_list_path = args.gene_list
    )
    processor.process()
    processor.save_to_h5(args.output_file, condition_keys=args.condition_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process single-cell RNA-seq data.")

    parser.add_argument(
        "--input_file",
        type=str,
        default="Global_raw.h5ad",
        help="Path to the input .h5ad file.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="processed_scrna_data.hdf5",
        help="Path to save the processed .hdf5 output file.",
    )

    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=500,
        help="Number of highly variable genes to select.",
    )

    parser.add_argument(
        "--min_cells",
        type=int,
        default=100000,
        help="Minimum number of cells a gene must be expressed in.",
    )

    parser.add_argument(
        "--condition_keys",
        nargs="+",
        default=["cell_type", "batch"],
        help="List of column names from .obs to use as conditions.",
    )

    parser.add_argument(
        "--test_size", type=float, default=0.1, help="Fraction of the data to use for the test set."
    )
    parser.add_argument(
        "--val_size", type=float, default=0.1, help="Fraction of the data to use for the validation set."
    )

    parser.add_argument(
        "--is_count", type = bool, default = True, help = "Boolean of whether input h5ad stores raw counts or processed values."
    )

    parser.add_argument(
        "--gene_list", type = str, default = None, help = "path file to a list of genes to keep"
    )
    args = parser.parse_args()
    main(args)
