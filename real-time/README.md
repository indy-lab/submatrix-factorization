# Real-Time Predictions

## Get the Data

First, [get the data](https://zenodo.org/record/3984924) from Zenodo.
Download the two zip files and put them in a folder of your choice (e.g., in a `../data/real-time/` folder).

Second, make sure you have access to historical data from the root of this repo in `/data/munvotesinfo.pkl`.

## Generate Predictions

Run the [generate-predictions.ipynb](generate-predictions.ipynb) notebook to generate predictions.

## Plot Predictions

Run the corresponding script in the [figures](../figures) folder:
~~~
python real-time-predictions.py ../data/real-time/predictions.pkl output/to/real-time-predictions.pdf
~~~
