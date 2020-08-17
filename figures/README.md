# Generate Figures

Each script in this folder takes two arguments:
1. Some data as input
2. A path to a PDF file as output

## Performance

To reproduce the performance plots for Switzerland, the US, and Germany, use the following scripts:
- `python ch-results.py ../data/testCH.pkl output/to/ch-results.pdf`
- `python us-results.py ../data/testUS.pkl output/to/us-results.pdf`
- `python de-results.py ../data/testDElocal5.pkl ../data/testDEstate9.pkl output/to/de-district-results.pdf` (This one takes two data files.)

The data files `test*.pkl` are the outputs of the `run_benchmark.py` script.
For example, `testUS.pkl` is the output of `python run_benchmark.py --name testUS ...` (see [README.md](../README.md)).

## Projections

To reproduce the projection plots (SVD and t-SNE) for Switzerland and Germany, use the following scripts:
- `python ch-svd.py ../data/projection/ch-votes.pkl output/to/ch-svd.pdf`
- `python ch-tsne.py ../data/projection/ch-votes.pkl output/to/ch-tsne.pdf`
- `python de-svd.py  ../data/ output/to/de-svd.pdf --embedding ../data/projection/de-embedding.pkl` (The `--embedding` enables to reuse the embedding generated from the script; they are automatically saved in the path of the data the first time you run it without the `--embedding` argument; the `data` argument is ignored when `embedding is given.)

The `ch-votes.pkl` file contains the historical vote results together with metadata about each municipality's language and canton.
The `de-embedding.pkl` file contains the precomputed t-SNE embedding for the German results at the district level.
You can regenerate it by running `python de-svd.py  ../data/ output/to/de-svd.pdf` (the `de-embedding.pkl` file will be saved in `../data/`.)

## Real-Time

See the [`real-time`](../real-time) folder for how to generate and plot these data.
