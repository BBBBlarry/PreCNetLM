echo "Task: Repeating Single Digit"
echo "PreCNetLM"
python simple_experiment.py -m "repeat_any" -v 10 -e 50 --num_stacks 3 --a_hat_size 64 --a_hat_layers 1 --r_size 32 --r_layers 2
echo "LSTM"
python simple_experiment_sota.py -m "repeat_any" -v 10 -e 50 --hidden_size 64 --num_layers 2

echo
echo "Task: Sequence of Numbers"
echo "PreCNetLM"
python simple_experiment.py -m "sequence" -v 10 -e 50 --num_stacks 3 --a_hat_size 64 --a_hat_layers 1 --r_size 32 --r_layers 2
echo "LSTM"
python simple_experiment_sota.py -m "sequence" -v 10 -e 50 --hidden_size 64 --num_layers 2

echo
echo "Task: Sequence of Repeating Numbers"
echo "PreCNetLM"
python simple_experiment.py -m "sequence_double" -v 10 -e 50 --num_stacks 3 --a_hat_size 64 --a_hat_layers 1 --r_size 32 --r_layers 2
echo "LSTM"
python simple_experiment_sota.py -m "sequence_double" -v 10 -e 50 --hidden_size 64 --num_layers 2

echo
echo "Task: Binary Counting"
echo "PreCNetLM"
python simple_experiment.py -m "binary_count" -v 10 -e 50 --num_stacks 3 --a_hat_size 64 --a_hat_layers 1 --r_size 32 --r_layers 2
echo "LSTM"
python simple_experiment_sota.py -m "binary_count" -v 10 -e 50 --hidden_size 64 --num_layers 2
