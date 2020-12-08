echo "Task: Repeating Single Digit"
echo "with upper level penalization"
python simple_experiment.py -m "repeat_any" -v 10 -e 50 --penalize_upper_levels
echo "no upper level penalization"
python simple_experiment.py -m "repeat_any" -v 10 -e 50

echo
echo "Task: Sequence of Numbers"
echo "with upper level penalization"
python simple_experiment.py -m "sequence" -v 10 -e 50 --penalize_upper_levels
echo "no upper level penalization"
python simple_experiment.py -m "sequence" -v 10 -e 50

echo
echo "Task: Sequence of Repeating Numbers"
echo "with upper level penalization"
python simple_experiment.py -m "sequence_double" -v 10 -e 50 --penalize_upper_levels
echo "no upper level penalization"
python simple_experiment.py -m "sequence_double" -v 10 -e 50

echo
echo "Task: Binary Counting"
echo "with upper level penalization"
python simple_experiment.py -m "binary_count" -v 10 -e 50 --penalize_upper_levels
echo "no upper level penalization"
python simple_experiment.py -m "binary_count" -v 10 -e 50
