import numpy as np
import pandas as pd

from torch.utils import data

from sklearn import preprocessing


class JASMIN(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        x_vars: list = None,
        t_var: str = "tot_aod",
        y_vars: list = None,
        t_bins: int = 2,
    ) -> None:
        super(JASMIN, self).__init__()
        # Handle default values
        if x_vars is None:
            x_vars = [
                "RH900",
                "RH850",
                "RH700",
                "LTS",
                "EIS",
                "w500",
                "whoi_sst",
            ]
        if y_vars is None:
            y_vars = ["l_re", "liq_pc", "cod", "cwp"]
        # Read csv
        df = pd.read_csv(root, index_col=0)
        # Filter AOD and Precip values
        df = df[df.tot_aod.between(0.07, 1.0)]
        df = df[df.precip < 0.5]
        # Make train test valid split
        days = df["timestamp"].unique()
        days_valid = set(days[5::7])
        days_test = set(days[6::7])
        days_train = set(days).difference(days_valid.union(days_test))
        # Fit preprocessing transforms
        df_train = df[df["timestamp"].isin(days_train)]
        self.data_xfm = preprocessing.StandardScaler()
        self.data_xfm.fit(df_train[x_vars].to_numpy())
        self.treatments_xfm = preprocessing.KBinsDiscretizer(
            n_bins=t_bins, encode="onehot-dense"
        )
        self.treatments_xfm.fit(df_train[t_var].to_numpy().reshape(-1, 1))
        self.targets_xfm = preprocessing.StandardScaler()
        self.targets_xfm.fit(df_train[y_vars].to_numpy())
        # Split the data
        if split == "train":
            _df = df[df["timestamp"].isin(days_train)]
        elif split == "valid":
            _df = df[df["timestamp"].isin(days_valid)]
        elif split == "test":
            _df = df[df["timestamp"].isin(days_test)]
        # Set variables
        self.data = self.data_xfm.transform(_df[x_vars].to_numpy(dtype="float32"))
        self.treatments = self.treatments_xfm.transform(
            _df[t_var].to_numpy(dtype="float32").reshape(-1, 1)
        )[:, 1:]
        self.targets = self.targets_xfm.transform(_df[y_vars].to_numpy(dtype="float32"))
        # Variable properties
        self.dim_input = self.data.shape[-1]
        self.dim_targets = self.targets.shape[-1]
        self.dim_treatments = t_bins - 1
        self.data_names = x_vars
        self.target_names = y_vars
        self.treatment_names = [t_var]

    @property
    def data_frame(self):
        data = np.hstack(
            [
                self.data_xfm.inverse_transform(self.data),
                self.treatments,
                self.targets_xfm.inverse_transform(self.targets),
            ],
        )
        return pd.DataFrame(
            data=data,
            columns=self.data_names + self.treatment_names + self.target_names,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> data.dataset.T_co:
        return self.data[index], self.treatments[index], self.targets[index]
