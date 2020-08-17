python run_benchmark.py --name trainCH --data_dir ../data/ --n_obs 275 --v_max 300 --models train --dataset CH --n_orders 10 --n_quantiles 50 --logscale
python run_benchmark.py --name trainUS --data_dir ../data/ --n_obs 8 --v_max 10 --models train --dataset US --n_orders 1000 --n_quantiles 25 --logscale
python run_benchmark.py --name trainDElocal --data_dir ../data/ --n_obs 3 --n_orders 100 --dataset DELocal --models train --v_max 4 --n_quantiles 50 --logscale
python run_benchmark.py --name trainDEstate --data_dir ../data/ --n_obs 4 --n_orders 100 --dataset DEState --models train --v_max 5 --n_quantiles 16
