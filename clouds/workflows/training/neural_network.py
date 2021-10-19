import json
from pathlib import Path

from clouds import models
from clouds import datasets


def train(
    config, experiment_dir, ensemble_id,
):
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

    experiment_dir = (
        Path(experiment_dir)
        / f"dh-{dim_hidden}_nc-{num_components}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_path = experiment_dir / "config.json"
    with config_path.open(mode="w") as cp:
        json.dump(config, cp)

    out_dir = experiment_dir / "checkpoints" / f"model-{ensemble_id}" / "mu"
    if not (out_dir / "best_checkpoint.pt").exists():
        outcome_model = models.MultiTaskNeuralNetwork(
            job_dir=out_dir,
            architecture="resnet",
            dim_input=train_dataset.dim_input,
            dim_treatment=train_dataset.dim_treatments,
            dim_output=train_dataset.dim_targets,
            dim_hidden=dim_hidden,
            num_components=num_components,
            depth=depth,
            negative_slope=negative_slope,
            layer_norm=False,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            num_examples=len(train_dataset),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=epochs,
            num_workers=0,
            seed=ensemble_id,
        )
        _ = outcome_model.fit(train_dataset=train_dataset, tune_dataset=valid_dataset)
    return -1
