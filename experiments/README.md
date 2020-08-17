# Experiments

## Data

We evaluate our approach and analyze the following four data sets:
1. Swiss referenda (326 referenda with binary outcome)
2. Popular vote of US presidential elections (binary outcomes between the Democrat and Republican candidates)
3. State-level German legislative elections (for the 5 major parties)
4. District-level (*Wahlkreise*) German legislative elections (for the 5 major parties)

These datasets are located in the [`data`](../data).

## Reproduce the Results

To reproduce the results and figures of the paper, simply run the script `scripts/test.sh`:

```bash
sh scripts/test.sh
```

This will train the model on all but the last vote (either referendum or election depending on the dataset), and it will evaluate on the last vote using the best hyperparameters we reported.

## Model Selection

Each experiment is split into three steps:
1. We run a grid search for the hyperparameters on historical data (`run_benchmark.py --models train`).
2. We select the best hyperparameters (`evaluate_benchmark.py`).
3. We evaluate the models using the best hyperparameters (`run_benchmark.py --models test`).

For Step 1, the ranges of hyperparameters are specified directly in `run_benchmark.py` (see `get_models() on lines 47-76).
the arguments to `run_benchmark.py` are as follows:
The arguments are as follows:
- `--name` is the filename of the results
- `--data_dir` is the directory for the dataset
- `--n_obs` is the number of observed (historical) votes
- `--v_max` is the last vote for which we make predictions
- `--models` is either _train_ or _test_ to select which procedure to run
- `--dataset` is one of _CH, US, DEState, DELocal_
- `--n_orders` is the number of randomized reveal-orders of regional results
- `--n_quantiles` controls at how many discrete points performance is measured
- `--logscale` enables selection of the discrete point on a logarithmic scale
For example, we run the following command the select the hyperparameters for the Swiss referenda dataset:
```bash
python run_benchmark.py --name trainCH --data_dir ../data/ --n_obs 275 --v_max 300 --models train --dataset CH --n_orders 10 --n_quantiles 50 --logscale
```

For Step 2, we run the following command to evaluate the results of the experiments to select the best hyperparameters:
```bash
python evaluate_benchmark.py --name trainCH
```
This will display the mean absolute errors for each model.
Here, `--name` should be the same as `--name` from Step 1.
We select the best combination of hyperparameters, and we add them to `run_benchmark.py` (see `get_models()` on lines 15-45)

Finally, for Step 3, we evaluate the model on a test set using the best hyperparameters from Step 2 using:
Then, we are ready to run the final benchmark using
```bash
python run_benchmark.py --name testCH --data_dir ../data/ --n_obs 300 --v_max 326 --models test --dataset CH --n_orders 100 --n_quantiles 50 --logscale
```

In short, to reproduce all results:
1. Run `scripts/train.sh`
2. Run `python evaluate_benchmark.py --name NAME` on all resulting files
3. Optionally adjust the test models in `run_benchmark.py`
4. Run `scripts/test.sh`.
