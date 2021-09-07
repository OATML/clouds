import ray
import click

from torch import cuda

from pathlib import Path


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
            "gpu_per_trial": gpu_per_model,
            "seed": seed,
            "tune": False,
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


if __name__ == "__main__":
    cli()
