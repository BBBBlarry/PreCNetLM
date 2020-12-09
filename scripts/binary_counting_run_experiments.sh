echo "Baseline"
python simple_experiment.py -m "binary_count" -v 32 -e 100 --num_stacks 4 --a_hat_size 64 --a_hat_layers 1 --r_size 64 --r_layers 1

echo "R layers experiments"
python simple_experiment.py -m "binary_count" -v 32 -e 100 --num_stacks 4 --a_hat_size 64 --a_hat_layers 1 --r_size 64 --r_layers 10

echo "R size experiments"
python simple_experiment.py -m "binary_count" -v 32 -e 100 --num_stacks 4 --a_hat_size 64 --a_hat_layers 1 --r_size 512 --r_layers 1

echo "A Hat layers experiments"
python simple_experiment.py -m "binary_count" -v 32 -e 100 --num_stacks 4 --a_hat_size 64 --a_hat_layers 10 --r_size 64 --r_layers 1

echo "A Hat size experiments"
python simple_experiment.py -m "binary_count" -v 32 -e 100 --num_stacks 4 --a_hat_size 512 --a_hat_layers 1 --r_size 64 --r_layers 1

echo "Number of stacks experiments"
python simple_experiment.py -m "binary_count" -v 32 -e 100 --num_stacks 16 --a_hat_size 64 --a_hat_layers 1 --r_size 64 --r_layers 1