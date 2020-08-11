# Sub-Matrix Factorization for Real-Time Vote Prediction

This repository contains the data and code to reproduce the experiments of

> Immer, A.\*, Kristof, V.\*, Grossglauser, M., Thiran, P., [*Sub-Matrix Factorization for Real-Time Vote Prediction*](https://infoscience.epfl.ch/record/278872), KDD 2020

For a standalone version of the algorithm propose in the paper, have a look at the [`predikon`](https://github.com/indy-lab/predikon) library.

## Data

We evaluate our approach and analyze the following 4 data sets:
1. 326 Swiss referenda (binary outcome)
2. public vote of US presidential elections (binary democrat vs. republican)
3. state-level German parliamentary elections (5 major parties)
4. _Wahlkreis_-level German parliamentary elections (5 major parties)

## Experiments

### Reproduce
To reproduce the results and figures in the paper, simply run the script `scripts/test.sh`.
This will perform the benchmark on the latest elections in all respective datasets.

### Reproduce model selection

Each experiment is split into three phases:
first, we assess potential hyperparameters on historical data (`run_benchmark.py --models train`).
Next, we select the best performing hyperparameters (`evaluate_benchmark.py`).
Finally, we re-run the best performing models (`run_benchmark.py --models test`).

Many more hyperparameters can be specified to customize experiments for other data sets.
As an example, consider the experiment on Swiss referenda.
In this case, we run the following command to run experiments for hyperparameter selection.
```bash
python run_benchmark.py --name trainCH --data_dir data/ --n_obs 275 --v_max 300 --models train --dataset CH --n_orders 10 --n_quantiles 50 --logscale
```
Here, `--name` indicates the filename of the results, `--data_dir` the directory containing all
necessary files to load data sets, `--n_obs` the amount of assumed previously observed votes,
`--v_max` the last vote to predict, `--models` either _train_ or _test_ to select from two model
sets, `--dataset` is one of _CH, US, DEState, DELocal_, `--n_orders` determines the number of
randomized reveal-orders of regional results, `--n_quantiles` controls at how many discrete points
performance is measured, and `--logscale` enables selection of the discrete point in a log scale.

For the second step, we need to evaluate the results of the experiments to select the
best-performing models. To do so, run
```bash
python evaluate_benchmark.py --name trainUS
```
which will yield a list of models per model class and the corresponding mean average errors.
We select the best performing models and specify them in `run_benchmark.py`.
Then, we are ready to run the final benchmark using
```bash
python run_benchmark.py --name testCH --data_dir data/ --n_obs 300 --v_max 326 --models test --dataset CH --n_orders 100 --n_quantiles 50 --logscale
```

In short, first run `scripts/train.sh`, then use `python evaluate_benchmark.py --name NAME` on all
resulting files. Potentially, adjust the test-models in `run_benchmark.py` and finally run
`scripts/test.sh`.

## Figures

## References

- [German parliamentary elections data set (1990-2009)](https://nsd.no/european_election_database/country/germany/parliamentary_elections.html)
- [US presidential election data set (1976-2016)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/42MVDX)
- [Swiss referenda data set (1981-present)](https://opendata.swiss/en/dataset/echtzeitdaten-am-abstimmungstag-zu-eidgenoessischen-abstimmungsvorlagen)

