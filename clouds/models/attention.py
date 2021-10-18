import torch

from torch import nn
from torch import optim
from torch.utils import data

from ignite import metrics

from clouds.models import core
from clouds.models import modules
from clouds.metrics.regression import NegR2Score


class MultiTaskAttentionNetwork(core.PyTorchModel):
    def __init__(
        self,
        job_dir,
        dim_input,
        dim_treatment,
        dim_output,
        num_components,
        dim_hidden,
        depth,
        num_heads,
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
        super(MultiTaskAttentionNetwork, self).__init__(
            job_dir=job_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            num_workers=num_workers,
        )
        encoder = modules.Encoder(
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            depth=depth,
            num_heads=num_heads,
            layer_norm=layer_norm,
            negative_slope=negative_slope,
            dropout_rate=dropout_rate,
            spectral_norm=spectral_norm,
        )
        hypothesis_0 = nn.Sequential()
        hypothesis_1 = nn.Sequential()
        for i in range(depth):
            hypothesis_0.add_module(
                f"hypothesis-0_{i}",
                modules.DenseResidual(
                    dim_input=dim_hidden,
                    dim_output=dim_hidden,
                    bias=not layer_norm,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    layer_norm=layer_norm,
                    spectral_norm=spectral_norm,
                ),
            )
            hypothesis_1.add_module(
                f"hypothesis-1_{i}",
                modules.DenseResidual(
                    dim_input=dim_hidden,
                    dim_output=dim_hidden,
                    bias=not layer_norm,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    layer_norm=layer_norm,
                    spectral_norm=spectral_norm,
                ),
            )
        hypothesis_0.add_module(
            "hypothesis-0_activation",
            modules.DenseActivation(
                dim_input=dim_hidden,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
            ),
        )
        hypothesis_1.add_module(
            "hypothesis-1_activation",
            modules.DenseActivation(
                dim_input=dim_hidden,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
            ),
        )
        treatment_density = modules.Categorical(
            dim_input=dim_hidden, dim_output=dim_treatment,
        )
        targets_density = modules.SplitGMM(
            num_components=num_components, dim_input=dim_hidden, dim_output=dim_output,
        )
        self.network_t = modules.DensityAttentionNetwork(
            encoder=encoder, density=treatment_density,
        )
        self.network_y = modules.TarAttentionNetwork(
            encoder=encoder,
            hypothesis_0=hypothesis_0,
            hypothesis_1=hypothesis_1,
            density=targets_density,
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

    def train_step_y(
        self, inputs, treatments, targets, position, inputs_mask,
    ):
        self.network_y.train()
        self.optimizer_y.zero_grad()
        outputs = self.network_y(
            inputs=inputs,
            treatments=treatments,
            position=position,
            inputs_mask=inputs_mask,
        )
        targets = targets.reshape(-1, targets.shape[-1])[inputs_mask.reshape(-1)]
        loss = -outputs.log_prob(targets).mean()
        loss.backward()
        self.optimizer_y.step()
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def tune_step_y(
        self, inputs, treatments, targets, position, inputs_mask,
    ):
        self.network_y.eval()
        with torch.no_grad():
            outputs = self.network_y(
                inputs=inputs,
                treatments=treatments,
                position=position,
                inputs_mask=inputs_mask,
            )
        targets = targets.reshape(-1, targets.shape[-1])[inputs_mask.reshape(-1)]
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def train_step_t(
        self, inputs, treatments, position, inputs_mask,
    ):
        self.network_t.train()
        self.optimizer_t.zero_grad()
        outputs = self.network_t(
            inputs=inputs,
            outputs=treatments,
            position=position,
            inputs_mask=inputs_mask,
        )
        treatments = treatments.reshape(-1, treatments.shape[-1])[
            inputs_mask.reshape(-1)
        ]
        loss = -outputs.log_prob(treatments).mean()
        loss.backward()
        self.optimizer_t.step()
        metric_values = {
            "outputs": outputs,
            "targets": treatments,
        }
        return metric_values

    def tune_step_t(
        self, inputs, treatments, position, inputs_mask,
    ):
        self.network_t.eval()
        with torch.no_grad():
            outputs = self.network_t(
                inputs=inputs,
                outputs=treatments,
                position=position,
                inputs_mask=inputs_mask,
            )
        treatments = treatments.reshape(-1, treatments.shape[-1])[
            inputs_mask.reshape(-1)
        ]
        metric_values = {
            "outputs": outputs,
            "targets": treatments,
        }
        return metric_values

    def train_step(self, engine, batch):
        inputs, treatments, targets, position = self.preprocess(batch)
        inputs_mask = inputs[:, :, :1].isnan().transpose(-2, -1) == False
        inputs[inputs.isnan()] = 0.0
        treatments[treatments.isnan()] = 0.0
        targets[targets.isnan()] = 0.0
        position[position.isnan()] = 0.0
        metrics_y = self.train_step_y(
            inputs=inputs,
            treatments=treatments,
            targets=targets,
            position=position,
            inputs_mask=inputs_mask,
        )
        metrics_t = self.train_step_t(
            inputs=inputs,
            treatments=treatments,
            position=position,
            inputs_mask=inputs_mask,
        )
        metric_values = {
            "y": metrics_y,
            "t": metrics_t,
        }
        return metric_values

    def tune_step(self, engine, batch):
        inputs, treatments, targets, position = self.preprocess(batch)
        inputs_mask = inputs[:, :, :1].isnan().transpose(-2, -1) == False
        inputs[inputs.isnan()] = 0.0
        treatments[treatments.isnan()] = 0.0
        targets[targets.isnan()] = 0.0
        position[position.isnan()] = 0.0
        metrics_y = self.tune_step_y(
            inputs=inputs,
            treatments=treatments,
            targets=targets,
            position=position,
            inputs_mask=inputs_mask,
        )
        metrics_t = self.tune_step_t(
            inputs=inputs,
            treatments=treatments,
            position=position,
            inputs_mask=inputs_mask,
        )
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
