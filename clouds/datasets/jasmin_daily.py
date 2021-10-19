import numpy as np
import pandas as pd

from torch.utils import data

from sklearn import preprocessing


class JASMINDaily(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        x_vars: list = None,
        t_var: str = "tot_aod",
        y_vars: list = None,
        t_bins: int = 2,
        pad: bool = False,
    ) -> None:
        super(JASMINDaily, self).__init__()
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
        _df = _df.groupby("timestamp")
        # Set variables
        self.covariates = []
        self.treatments = []
        self.targets = []
        self.position = []
        for _, group in _df:
            if len(group) > 1:
                self.covariates.append(
                    self.data_xfm.transform(group[x_vars].to_numpy(dtype="float32"))
                )
                self.treatments.append(
                    self.treatments_xfm.transform(
                        group[t_var].to_numpy(dtype="float32").reshape(-1, 1)
                    )[:, 1:]
                )
                self.targets.append(
                    self.targets_xfm.transform(group[y_vars].to_numpy(dtype="float32"))
                )
                self.position.append(group[["lats", "lons"]].to_numpy(dtype="float32"))
        # Variable properties
        self.dim_input = self.covariates[0].shape[-1]
        self.dim_targets = self.targets[0].shape[-1]
        self.dim_treatments = t_bins - 1
        self.data_names = x_vars
        self.target_names = y_vars
        self.treatment_names = [t_var]
        self.split = split
        self.pad = pad

    @property
    def data_frame(self):
        data = np.hstack(
            [
                self.data_xfm.inverse_transform(np.vstack(self.covariates)),
                np.vstack(self.treatments),
                self.targets_xfm.inverse_transform(np.vstack(self.targets)),
            ],
        )
        return pd.DataFrame(
            data=data,
            columns=self.data_names + self.treatment_names + self.target_names,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> data.dataset.T_co:
        covariates = self.covariates[index]
        treatments = self.treatments[index]
        targets = self.targets[index]
        position = self.position[index]
        if self.pad:
            num_samples = covariates.shape[0]
            if num_samples < 80:
                diff = 80 - num_samples
                pading = ((0, diff), (0, 0))
                covariates = np.pad(covariates, pading, constant_values=np.nan)
                treatments = np.pad(treatments, pading, constant_values=np.nan)
                targets = np.pad(targets, pading, constant_values=np.nan)
                position = np.pad(position, pading, constant_values=np.nan)
            elif num_samples > 80:
                sample = np.random.choice(np.arange(num_samples), 80, replace=False)
                sample.sort()
                covariates = covariates[sample]
                treatments = treatments[sample]
                targets = targets[sample]
                position = position[sample]
        position -= position.mean(0)
        if self.split == "train" and self.pad:
            position += np.random.uniform(-10, 10, size=(1, 2))
        return covariates, treatments, targets, position
