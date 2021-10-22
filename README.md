# clouds

Exploring the effects of aerosols on proxies for cloud reflectivity

## Installation

```.sh
git clone git@github.com:anndvision/clouds.git
cd clouds
conda env create -f environment.yml
conda activate clouds
```

[Optional] Developer mode

```.sh
pip install -e .
```

## Make Data

Right now we just copy Peter's dataset locally. Creation from scratch to come.

```.sh
mkdir data
cp [wherever you have it]/four_outputs_liqcf_pacific.csv data/
```

## Quince Model

### Train model

First we will need to train a model. This can be done using the following command:

```.sh
clouds train --job-dir output/ --gpu-per-model 0.25 jasmin --root data/four_outputs_liqcf_pacific.csv ensemble
```

In this example we write to the directory `output/`, which will be created wherever you run the above command. The model parameters will be saved in this directory. You can change the job-dir directory to anywhere you would rather write files to.

### Make predictions

Now we load the trained model and predict the CATE with it. 

```.sh
clouds predict --job-dir output/ --gpu-per-model 0.25 jasmin --root data/four_outputs_liqcf_pacific.csv ensemble
```

This command will write 3 csv files. One corresponding to the train, valid, and test sets:

`output/jasmin/ensemble/dh-800_nc-20_dp-3_ns-0.0_dr-0.5_sn-0.0_lr-0.0002_bs-4096_ep-400/results_train.csv`

`output/jasmin/ensemble/dh-800_nc-20_dp-3_ns-0.0_dr-0.5_sn-0.0_lr-0.0002_bs-4096_ep-400/results_valid.csv`

`output/jasmin/ensemble/dh-800_nc-20_dp-3_ns-0.0_dr-0.5_sn-0.0_lr-0.0002_bs-4096_ep-400/results_test.csv`

### Plot results

Now we plot the results:

Training set

```.sh
clouds plot --csv-path output/jasmin/ensemble/dh-800_nc-20_dp-3_ns-0.0_dr-0.5_sn-0.0_lr-0.0002_bs-4096_ep-400/results_train.csv --output-dir output/ensemble/train/
```

Validation set

```.sh
clouds plot --csv-path output/jasmin/ensemble/dh-800_nc-20_dp-3_ns-0.0_dr-0.5_sn-0.0_lr-0.0002_bs-4096_ep-400/results_valid.csv --output-dir output/ensemble/valid/
```

Test set

```.sh
clouds plot --csv-path output/jasmin/ensemble/dh-800_nc-20_dp-3_ns-0.0_dr-0.5_sn-0.0_lr-0.0002_bs-4096_ep-400/results_test.csv --output-dir output/ensemble/test/
```

You can now look in the folder `output/ensemble` and see the results.

### Hyperparameter Tuning

The above examples use the hyper-paramters determined by the following command:

```.sh
clouds tune --job-dir output/ --gpu-per-model 0.2 jasmin --root data/four_outputs_liqcf_pacific.csv ensemble
```
