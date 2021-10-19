from pathlib import Path

from clouds import models
from clouds import datasets


def predict(
    config, experiment_dir,
):
    dataset_name = config.get("dataset_name")
    train_dataset = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    valid_dataset = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))
    test_dataset = datasets.DATASETS.get(dataset_name)(**config.get("ds_test"))

    train_dataframe = train_dataset.data_frame
    valid_dataframe = valid_dataset.data_frame
    test_dataframe = test_dataset.data_frame

    dim_hidden = config.get("dim_hidden")
    num_components = config.get("num_components")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    ensemble_size = config.get("ensemble_size")

    experiment_dir = (
        Path(experiment_dir)
        / f"dh-{dim_hidden}_nc-{num_components}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
    )
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file {config_path} does not exist, are you sure you have specified the right --job-dir or model parameterrs?"
        )

    for ensemble_id in range(ensemble_size):
        out_dir = experiment_dir / "checkpoints" / f"model-{ensemble_id}" / "mu"
        model = models.MultiTaskNeuralNetwork(
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
        model.load()
        train_dataframe = update_dataframe(
            model=model,
            dataset=train_dataset,
            dataframe=train_dataframe,
            ensemble_id=ensemble_id,
        )
        valid_dataframe = update_dataframe(
            model=model,
            dataset=valid_dataset,
            dataframe=valid_dataframe,
            ensemble_id=ensemble_id,
        )
        test_dataframe = update_dataframe(
            model=model,
            dataset=test_dataset,
            dataframe=test_dataframe,
            ensemble_id=ensemble_id,
        )
    train_dataframe.to_csv(experiment_dir / "results_train.csv")
    valid_dataframe.to_csv(experiment_dir / "results_valid.csv")
    test_dataframe.to_csv(experiment_dir / "results_test.csv")
    return -1


def update_dataframe(model, dataset, dataframe, ensemble_id):
    tau = model.predict_tau(dataset)
    y = model.predict_y_mean(dataset)
    for i, label in enumerate(dataset.target_names):
        dataframe[f"CATE {label} {ensemble_id}"] = tau[:, i]
        dataframe[f"Y {label} {ensemble_id}"] = y[:, i]
    dataframe = update_summary_stats(dataset, dataframe,)
    return dataframe


def update_summary_stats(dataset, dataframe):
    for i, label in enumerate(dataset.target_names):
        dataframe[f"Predicted {TARGET_KEYS[label]}"] = dataframe[
            [c for c in dataframe.columns if f"Y {label}" in c]
        ].mean(1)
        dataframe[f"Observed {TARGET_KEYS[label]}"] = dataframe[f"{label}"]
        dataframe[f"CATE {TARGET_KEYS[label]}"] = dataframe[
            [c for c in dataframe.columns if f"CATE {label}" in c]
        ].mean(1)
        dataframe[f"CATE Uncertainty {TARGET_KEYS[label]}"] = 2 * dataframe[
            [c for c in dataframe.columns if f"CATE {label}" in c]
        ].std(1)
    return dataframe


TARGET_KEYS = {
    "l_re": r"$r_e$",
    "liq_pc": r"$CF_w$",
    "cod": r"$\tau$",
    "cwp": r"$LWP$",
}
