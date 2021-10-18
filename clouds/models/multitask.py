from sklearn.utils.extmath import density
import torch

from torch import nn
from torch import optim
from torch.utils import data

from ignite import metrics

from clouds.models import core
from clouds.models import modules
from clouds.metrics.regression import NegR2Score


class MultiTaskNeuralNetwork(core.PyTorchModel):
    def __init__(
        self,
        job_dir,
        architecture,
        dim_input,
        dim_treatment,
        dim_output,
        num_components,
        dim_hidden,
        depth,
        negative_slope,
        layer_norm,
        spectral_norm,
        dropout_rate,
        num_examples,
        learning_rate,
        batch_size,
        epochs,
        patience,
        num_workers,
        seed,
    ):
        super(MultiTaskNeuralNetwork, self).__init__(
            job_dir=job_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
        )
        feature_extractor = nn.Sequential(
            modules.DenseLinear(
                dim_input=dim_input,
                dim_output=dim_hidden,
                layer_norm=layer_norm,
                spectral_norm=spectral_norm,
            ),
            modules.DenseFeatureExtractor(
                architecture=architecture,
                dim_input=dim_hidden,
                dim_hidden=dim_hidden,
                depth=depth - 1,
                negative_slope=negative_slope,
                layer_norm=layer_norm,
                dropout_rate=dropout_rate,
                spectral_norm=spectral_norm,
                activate_output=True,
            ),
        )
        self.network_t = modules.NeuralDensityNetwork(
            feature_extractor=feature_extractor,
            density_estimator=modules.Categorical(
                dim_input=dim_hidden, dim_output=dim_treatment,
            ),
        )
        self.network_y = modules.TarNet(
            feature_extractor=feature_extractor,
            hypothesis_0=nn.Identity(),
            hypothesis_1=nn.Identity(),
            density_estimator=modules.SplitGMM(
                num_components=num_components,
                dim_input=dim_hidden,
                dim_output=dim_output,
            ),
        )
        self.metrics = {
            "loss": NegR2Score(
                dim_output=dim_output,
                output_transform=lambda x: (x["y"]["outputs"].mean, x["y"]["targets"],),
                device=self.device,
            ),
            "loss_y": metrics.Average(
                output_transform=lambda x: -x["y"]["outputs"]
                .log_prob(x["y"]["targets"])
                .mean(),
                device=self.device,
            ),
            "loss_t": metrics.Average(
                output_transform=lambda x: -x["t"]["outputs"]
                .log_prob(x["t"]["targets"])
                .mean(),
                device=self.device,
            ),
        }
        self.batch_size = batch_size
        self.best_loss = 1e7
        self.patience = patience
        self.optimizer_t = optim.Adam(
            params=self.network_t.parameters(),
            lr=self.learning_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / num_examples,
        )
        self.optimizer_y = optim.Adam(
            params=self.network_y.parameters(),
            lr=self.learning_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / num_examples,
        )
        self.network_t.to(self.device)
        self.network_y.to(self.device)

    def train_step_y(self, inputs, treatments, targets):
        self.network_y.train()
        self.optimizer_y.zero_grad()
        outputs = self.network_y([inputs, treatments])
        loss = -outputs.log_prob(targets).mean()
        loss.backward()
        self.optimizer_y.step()
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def tune_step_y(self, inputs, treatments, targets):
        self.network_y.eval()
        with torch.no_grad():
            outputs = self.network_y([inputs, treatments])
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def train_step_t(self, inputs, targets):
        self.network_t.train()
        self.optimizer_t.zero_grad()
        outputs = self.network_t(inputs)
        loss = -outputs.log_prob(targets).mean()
        loss.backward()
        self.optimizer_t.step()
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def tune_step_t(self, inputs, targets):
        self.network_t.eval()
        with torch.no_grad():
            outputs = self.network_t(inputs)
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def train_step(self, engine, batch):
        inputs, treatments, targets = self.preprocess(batch)
        metrics_y = self.train_step_y(inputs, treatments, targets)
        metrics_t = self.train_step_t(inputs, treatments)
        metric_values = {
            "y": metrics_y,
            "t": metrics_t,
        }
        return metric_values

    def tune_step(self, engine, batch):
        inputs, treatments, targets = self.preprocess(batch)
        metrics_y = self.tune_step_y(inputs, treatments, targets)
        metrics_t = self.tune_step_t(inputs, treatments)
        metric_values = {
            "y": metrics_y,
            "t": metrics_t,
        }
        return metric_values

    def predict_tau(self, dataset):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        mu0 = []
        mu1 = []
        self.network_y.eval()
        with torch.no_grad():
            for batch in dl:
                inputs, treatments, _ = self.preprocess(batch)
                mu0.append(self.network_y([inputs, torch.zeros_like(treatments)]).mean)
                mu1.append(self.network_y([inputs, torch.ones_like(treatments)]).mean)
        if dataset.targets_xfm is not None:
            mu0 = dataset.targets_xfm.inverse_transform(
                torch.cat(mu0, dim=0).to("cpu").numpy()
            )
            mu1 = dataset.targets_xfm.inverse_transform(
                torch.cat(mu1, dim=0).to("cpu").numpy()
            )
        return mu1 - mu0

    def predict_y_mean(self, dataset):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        mean = []
        self.network_y.eval()
        with torch.no_grad():
            for batch in dl:
                inputs, treatments, _ = self.preprocess(batch)
                mean.append(self.network_y([inputs, treatments]).mean)
        if dataset.targets_xfm is not None:
            mean = dataset.targets_xfm.inverse_transform(
                torch.cat(mean, dim=0).to("cpu").numpy()
            )
        return mean

    def sample_y(self, dataset):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        y = []
        self.network_y.eval()
        with torch.no_grad():
            for batch in dl:
                inputs, treatments, _ = self.preprocess(batch)
                y.append(self.network_y([inputs, treatments]).sample())
        if dataset.targets_xfm is not None:
            y = dataset.targets_xfm.inverse_transform(
                torch.cat(y, dim=0).to("cpu").numpy()
            )
        return y
