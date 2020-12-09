echo "Baseline"
python simple_experiment.py -m "binary_count" -v 32 -e 100 --num_stacks 4 --a_hat_size 64

echo "R layers experiments"
python simple_experiment.py -m "binary_count" -v 32 -e 100 --r_layers 2
python simple_experiment.py -m "binary_count" -v 32 -e 100 --r_layers 8