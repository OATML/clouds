import ray
import click

from torch import cuda

from pathlib import Path

from clouds import workflows


@click.group(chain=True)
@click.pass_context
def cli(context):
    context.obj = {"n_gpu": cuda.device_count()}


@cli.command("train")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option(
    "--gpu-per-model",
    default=cuda.device_count(),
    type=float,
    help="number of gpus for each ensemble model, default=cuda.device_count()",
)
@click.option(
    "--seed", default=1331, type=int, help="random number generator seed, default=1331",
)
@click.pass_context
def train(
    context, job_dir, gpu_per_model, seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
    )
    context.obj.update(
        {
            "job_dir": job_dir,
            "gpu_per_model": gpu_per_model,
            "seed": seed,
            "mode": "train",
        }
    )


@cli.command("tune")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option(
    "--max-samples",
    default=500,
    type=int,
    help="maximum number of search space samples, default=100",
)
@click.option(
    "--gpu-per-model",
    default=cuda.device_count(),
    type=float,
    help="number of gpus for each ensemble model, default=cuda.device_count()",
)
@click.option(
    "--seed", default=1331, type=int, help="random number generator seed, default=1331",
)
@click.pass_context
def tune(
    context, job_dir, max_samples, gpu_per_model, seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
    )
    context.obj.update(
        {
            "job_dir": job_dir,
            "max_samples": max_samples,
            "gpu_per_model": gpu_per_model,
            "seed": seed,
            "mode": "tune",
        }
    )


@cli.command("predict")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option(
    "--gpu-per-model",
    default=cuda.device_count(),
    type=float,
    help="number of gpus for each ensemble model, default=cuda.device_count()",
)
@click.option(
    "--seed", default=1331, type=int, help="random number generator seed, default=1331",
)
@click.pass_context
def predict(
    context, job_dir, gpu_per_model, seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
    )
    context.obj.update(
        {
            "job_dir": job_dir,
            "gpu_per_model": gpu_per_model,
            "seed": seed,
            "mode": "predict",
        }
    )


@cli.command("jasmin")
@click.pass_context
@click.option(
    "--root", type=str, required=True, help="location of dataset",
)
def jasmin(
    context, root,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "jasmin"
    experiment_dir = job_dir / dataset_name
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {"root": root, "split": "train",},
            "ds_valid": {"root": root, "split": "valid",},
            "ds_test": {"root": root, "split": "test",},
        }
    )


@cli.command("ensemble")
@click.pass_context
@click.option("--dim-hidden", default=800, type=int, help="num neurons")
@click.option("--num-components", default=20, type=int, help="num mixture components")
@click.option("--depth", default=3, type=int, help="depth of feature extractor")
@click.option(
    "--negative-slope",
    default=0.0,
    type=float,
    help="negative slope of leaky relu, default=-1 use elu",
)
@click.option(
    "--dropout-rate", default=0.5, type=float, help="dropout rate, default=0.1"
)
@click.option(
    "--spectral-norm",
    default=0.0,
    type=float,
    help="Spectral normalization coefficient. If 0.0 do not use spectral norm, default=0.0",
)
@click.option(
    "--learning-rate",
    default=2e-4,
    type=float,
    help="learning rate for gradient descent, default=1e-3",
)
@click.option(
    "--batch-size",
    default=4096,
    type=int,
    help="number of examples to read during each training step, default=100",
)
@click.option(
    "--epochs", type=int, default=400, help="number of training epochs, default=400"
)
@click.option(
    "--ensemble-size",
    type=int,
    default=10,
    help="number of models in ensemble, default=1",
)
def ensemble(
    context,
    dim_hidden,
    num_components,
    depth,
    negative_slope,
    dropout_rate,
    spectral_norm,
    learning_rate,
    batch_size,
    epochs,
    ensemble_size,
):
    if context.obj["mode"] == "tune":
        context.obj.update(
            {"epochs": epochs, "ensemble_size": ensemble_size,}
        )
        workflows.tuning.hyper_tune(config=context.obj)
    else:
        context.obj.update(
            {
                "dim_hidden": dim_hidden,
                "depth": depth,
                "num_components": num_components,
                "negative_slope": negative_slope,
                "dropout_rate": dropout_rate,
                "spectral_norm": spectral_norm,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "ensemble_size": ensemble_size,
            }
        )

        if context.obj["mode"] == "train":

            @ray.remote(num_gpus=context.obj.get("gpu_per_model"),)
            def trainer(**kwargs):
                func = workflows.training.train_ensemble(**kwargs)
                return func

            results = []
            for ensemble_id in range(ensemble_size):
                results.append(
                    trainer.remote(
                        config=context.obj,
                        experiment_dir=context.obj.get("experiment_dir"),
                        ensemble_id=ensemble_id,
                    )
                )
            ray.get(results)
        elif context.obj["mode"] == "predict":

            workflows.prediction.predict_ensemble(
                config=context.obj, experiment_dir=context.obj.get("experiment_dir")
            )


if __name__ == "__main__":
    cli()
