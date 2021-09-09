import os
import copy
import torch

from abc import ABC

from ray import tune

from ignite import utils
from ignite import engine
from ignite import distributed

from torch.utils import data
from torch.utils import tensorboard


class BaseModel(ABC):
    def __init__(self, job_dir, seed):
        super(BaseModel, self).__init__()
        self.job_dir = job_dir
        self.seed = seed

    def fit(self, train_dataset, tune_dataset):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement train()"
        )

    def save(self):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement save()"
        )

    def load(self):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement load()"
        )


class PyTorchModel(BaseModel):
    def __init__(self, job_dir, learning_rate, batch_size, epochs, num_workers, seed):
        super(PyTorchModel, self).__init__(job_dir=job_dir, seed=seed)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.summary_writer = tensorboard.SummaryWriter(log_dir=self.job_dir)
        self.logger = utils.setup_logger(
            name=__name__ + "." + self.__class__.__name__, distributed_rank=0
        )
        self.trainer = engine.Engine(self.train_step)
        self.evaluator = engine.Engine(self.tune_step)
        self._network_t = None
        self._network_y = None
        self._optimizer_t = None
        self._optimizer_y = None
        self._metrics = None
        self.likelihood = None
        self.num_workers = num_workers
        self.device = distributed.device()
        self.best_state = None
        self.counter = 0

    @property
    def network_t(self):
        return self._network_t

    @network_t.setter
    def network_t(self, value):
        self._network_t = value

    @property
    def network_y(self):
        return self._network_y

    @network_y.setter
    def network_y(self, value):
        self._network_y = value

    @property
    def optimizer_t(self):
        return self._optimizer_t

    @optimizer_t.setter
    def optimizer_t(self, value):
        self._optimizer_t = value

    @property
    def optimizer_y(self):
        return self._optimizer_y

    @optimizer_y.setter
    def optimizer_y(self, value):
        self._optimizer_y = value

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    def train_step(self, trainer, batch):
        raise NotImplementedError()

    def tune_step(self, trainer, batch):
        raise NotImplementedError()

    def preprocess(self, batch):
        batch = (
            [x.to(self.device) for x in batch]
            if isinstance(batch, list)
            else batch.to(self.device)
        )
        return batch

    def fit(self, train_dataset, tune_dataset):
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        tune_loader = data.DataLoader(
            tune_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        )
        # Instantiate trainer
        for k, v in self.metrics.items():
            v.attach(self.trainer, k)
            v.attach(self.evaluator, k)
        self.trainer.add_event_handler(
            engine.Events.EPOCH_COMPLETED,
            self.on_epoch_completed,
            train_loader,
            tune_loader,
        )
        self.trainer.add_event_handler(
            engine.Events.COMPLETED, self.on_training_completed, tune_loader
        )
        # Train
        self.trainer.run(train_loader, max_epochs=self.epochs)
        return self.evaluator.state.metrics

    def on_epoch_completed(self, engine, train_loader, tune_loader):
        train_metrics = self.trainer.state.metrics
        print("Metrics Epoch", engine.state.epoch)
        justify = max(len(k) for k in train_metrics) + 2
        for k, v in train_metrics.items():
            if type(v) == float:
                print("train {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
            print("train {:<{justify}} {:<5}".format(k, v, justify=justify))
        self.evaluator.run(tune_loader)
        tune_metrics = self.evaluator.state.metrics
        if tune.is_session_enabled():
            tune.report(mean_loss=tune_metrics["loss"])
        justify = max(len(k) for k in tune_metrics) + 2
        for k, v in tune_metrics.items():
            if type(v) == float:
                print("tune {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
        if tune_metrics["loss"] < self.best_loss:
            self.best_loss = tune_metrics["loss"]
            self.counter = 0
            self.update()
        else:
            self.counter += 1
        if self.counter == self.patience:
            self.logger.info(
                "Early Stopping: No improvement for {} epochs".format(self.patience)
            )
            engine.terminate()

    def on_training_completed(self, engine, loader):
        if not tune.is_session_enabled():
            self.save()
            self.load()
            self.evaluator.run(loader)
            metric_values = self.evaluator.state.metrics
            print("Metrics Epoch", engine.state.epoch)
            justify = max(len(k) for k in metric_values) + 2
            for k, v in metric_values.items():
                if type(v) == float:
                    print("best {:<{justify}} {:<5f}".format(k, v, justify=justify))
                    continue

    def update(self):
        if not tune.is_session_enabled():
            self.best_state = {
                "network_t": copy.deepcopy(self.network_t.state_dict()),
                "network_y": copy.deepcopy(self.network_y.state_dict()),
                "optimizer_t": copy.deepcopy(self.optimizer_t.state_dict()),
                "optimizer_y": copy.deepcopy(self.optimizer_y.state_dict()),
                "engine": copy.copy(self.trainer.state),
            }
            if self.likelihood is not None:
                self.best_state["likelihood"] = copy.deepcopy(
                    self.likelihood.state_dict()
                )

    def save(self):
        p = os.path.join(self.job_dir, "best_checkpoint.pt")
        torch.save(self.best_state, p)

    def load(self):
        if tune.is_session_enabled():
            with tune.checkpoint_dir(step=self.trainer.state.epoch) as checkpoint_dir:
                p = os.path.join(checkpoint_dir, "checkpoint.pt")
        else:
            file_name = "best_checkpoint.pt"
            p = os.path.join(self.job_dir, file_name)
        if not os.path.exists(p):
            self.logger.info(
                "Checkpoint {} does not exist, starting a new engine".format(p)
            )
            return
        self.logger.info("Loading saved checkpoint {}".format(p))
        checkpoint = torch.load(p)
        self.network_t.load_state_dict(checkpoint["network_t"])
        self.network_y.load_state_dict(checkpoint["network_y"])
        self.optimizer_t.load_state_dict(checkpoint["optimizer_t"])
        self.optimizer_y.load_state_dict(checkpoint["optimizer_y"])
        if self.likelihood is not None:
            self.likelihood.load_state_dict(checkpoint["likelihood"])
        self.trainer.state = checkpoint["engine"]
