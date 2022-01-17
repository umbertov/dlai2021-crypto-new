# Cryptocurrency trend classification model

This is a rewrite of `https://github.com/umbertov/dlai2021-stonks`.

## Install dependencies
Preferrably set up a virtual environment beforehand.

The `requirements.txt` file contains only the explicitly installed dependencies, without exact version numbers. 

The output of `pip freeze` is instead saved in `exact_requirements.txt`.

```bash
pip install -r exact_requirements.txt
# or, if you want the latest packages:
pip install -r requirements.txt
```

## Data Gathering

```bash
cd data/ccxt_ohlcv
bash download_data.sh
```

## Training models

Just running
```bash
python src/run.py
```
is enough to train the LSTM model on Bitcoin and Ethereum 5-minutes data.

The other models in the report can be trained like so:
```bash
python -- src/run.py experiment="<one of lstm_large, tcn_wide_deep or linear_tagger >"
```

`conf/default.yaml` and the files in `conf/experiment` are already set up to use the configurations and hyperparams used for the models in the report.

The `conf/experiment` and `conf/model` folders contain various models which were used not only for the final classification task, but also other classification, regression and auto-encoder tasks. The non-classification models may not work due to breaking changes introduced after their development stopped (due to no success in the tasks at hand). 

The `conf/dataset_conf` includes many dataset configurations (features, objectives, normalizations, etc). The one used finally is `trend_fixclf_multivar.yaml`.

## Evaluating/Backtesting models

To evaluate a model, you just need its Weights & Biases run id. By default, my w&b project and entity are used, but you can change them with the `--project` and `--entity` command-line switches.

```
# Classification metrics
python -- src/evaluation/evaluate_classification.py --run-id WANDB_RUN_ID --use-split <train, val or test>

# Backtest
python -- src/evaluation/backtests.py --run-id WANDB_RUN_ID --use-split <train, val or test>
```

Use the `--help` switch or read the source files to view other arguments.

Evaluation for the report was done like so:
```bash
for split in train val test ; do 
    python -- src/evaluation/evaluate_classification.py WANDB_RUN_ID --use-split $split
done
for split in train val test ; do 
    python -- src/evaluation/backtests.py WANDB_RUN_ID --use-split $split
done
```

## Code Structure

Most of the project structure inherits from `lucmos/nn-template`.

#### Models

- `src/models/` contains the models that are NOT `LightningModule`s. Some of them, like `autoencoder.py`, `regression.py` and `vae.py` were not used for the report and their development was discontinued.
- `src/lightning_modules/` contains all the `LightningModule`s. During the experimental phase, many architectures and task settings were tried, including multi-task ones (e.g. regression + reconstruction, regression + classification, etc...), so there's a `base.py` file which defines a base model which is capable of handling regression, classification and auto-encoder tasks, and the task-specific implementations are plugged in from the task-specific files.
- `src/tcn.py` contains the code for the Temporal Convolutional Network from `https://github.com/locuslab/TCN/raw/master/TCN/tcn.py` with some additions that can be viewed with `git log --patch src/tcn.py`, or `git diff git diff df9b875 src/tcn.py` to just see the diff with the original version.
- `src/dyn_loss.py` is a modified version of `CrossEntropyLoss` which computes class weights (for the `weights` parameter of `torch.nn.functional.cross_entropy`) based on a running estimate of the frequency of each class. The estimate starts with equal weight for every class, and is update based on training data, smoothed with an exponential decay (i.e. `new_weight = [(1-decay)*old_weight] + [decay * new_weight]`, with `decay = 0.9` or `0.8`). Also, to prevent any of the weights from going to zero, a "minimum weight" parameter is used, empirically set to either `0.1` or `0.2`.




#### Data

The data is first read from the CSV files as a Pandas DataFrame with a common format (since data from CCXT and data from Yahoo! Finance use different column names and timestamp formats), and the relevant features and target columns are added to it.

Then the DataFrame is used to construct a PyTorch `Dataset` class instance. The `Dataset` class also handles splitting the time series into "windows" of fixed length, and some normalization schemes not used in the final report, like independently Z-Score or Min-Max scaling each window (which wouldn't be easily possible when working with a DataFrame).

- `src/dataset_readers.py` handles all the reading from CSV and feature computation. It contains a library of "Operations" which get a DataFrame as input, and produce another (altered) DataFrame as output. These operations are composable building blocks, and are used to build "`readers`". 
- `src/dataset.py` contains the `DataFrameDataset` class which supports regression, classification and auto-encoder tasks, as well as various functions to build a dataset from a CSV file using a `reader` from the `dataset_readers` module.
- `src/datamodule.py` is doesn't add much more than what was in `nn-template`.

#### Misc 

##### Streamlit apps

- `src/ui/categorical_dataviz.py` visualizes the categorical labels on the dataset, and optionally the predictions of a pretrained model (coming from a w&b run).
- `src/ui/ui.py` interactively visualizes the effect of running z-score normalization of the data when the lookback period changes.
- `src/evaluation/` contains evaluation and backtesting code.


#### Hydra conf

It is mostly the same as `nn-template`; the additions are primarily in `dataset_conf` and `model`.

