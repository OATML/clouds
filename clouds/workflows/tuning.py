from ray import tune
from ray.tune import schedulers
from ray.tune.suggest import bohb

from clouds import models
from clouds import datasets


def tune_multitask(config):
    dataset_name = config.get("dataset_name")
    train_dataset = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    valid_dataset = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

    dim_hidden = config.get("dim_hidden")
    num_components = config.get("num_components")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")

    outcome_model = models.MultiTaskNeuralNetwork(
        job_dir=None,
        architecture="resnet",
        dim_input=train_dataset.dim_input,
        dim_treatment=train_dataset.dim_treatments,
        dim_output=train_dataset.dim_targets,
        dim_hidden=dim_hidden,
        num_components=num_components,
        depth=depth,
        negative_slope=negative_slope,
        batch_norm=False,
        spectral_norm=spectral_norm,
        dropout_rate=dropout_rate,
        num_examples=len(train_dataset),
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=epochs,
        num_workers=0,
        seed=config.get("seed"),
    )
    _ = outcome_model.fit(train_dataset, valid_dataset)


def hyper_tune(config):
    space = {
        "dim_hidden": tune.choice([50, 100, 200, 400, 800]),
        "depth": tune.choice([2, 3, 4, 5]),
        "num_components": tune.choice([1, 2, 5, 10, 20]),
        "negative_slope": tune.choice([0.0, 0.1, 0.2, 0.3, -1.0]),
        "dropout_rate": tune.choice([0.0, 0.1, 0.2, 0.5]),
        "spectral_norm": tune.choice([0.0, 0.95, 1.5, 3.0]),
        "learning_rate": tune.choice([2e-4, 5e-4, 1e-3]),
        "batch_size": tune.choice([1024, 2048, 4096]),
    }
    algorithm = bohb.TuneBOHB(space, max_concurrent=5, metric="mean_loss", mode="min",)
    scheduler = schedulers.HyperBandForBOHB(
        time_attr="training_iteration", max_t=config.get("epochs"),
    )
    analysis = tune.run(
        run_or_experiment=tune_multitask,
        metric="mean_loss",
        mode="min",
        name="bohb",
        resources_per_trial={"gpu": config.get("gpu_per_model"),},
        num_samples=config.get("max_samples"),
        search_alg=algorithm,
        scheduler=scheduler,
        local_dir=config.get("experiment_dir"),
        config=config,
    )
    print("Best hyperparameters found were: ", analysis.best_config)
