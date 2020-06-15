python run_benchmark.py --name testCH --data_dir data/ --n_obs 300 --v_max 326 --models test --dataset CH --n_orders 100 --n_quantiles 50 --logscale
python run_benchmark.py --name testUS --data_dir data/ --n_obs 10 --v_max 11 --models test --dataset US --n_orders 1000 --n_quantiles 25 --logscale
python run_benchmark.py --name testDElocal --data_dir data/ --n_obs 4 --n_orders 1000 --dataset DELocal --models test --v_max 5 --n_quantiles 50 --logscale
python run_benchmark.py --name testDEstate --data_dir data/ --n_obs 5 --n_orders 1000 --dataset DEState --models test --v_max 6 --n_quantiles 16
