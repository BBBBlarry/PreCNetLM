echo "Task: Repeating Single Digit"
echo "with upper level penalization"
python simple_experiment.py -m "repeat_any" -v 40 -e 100 --penalize_upper_levels
echo "no upper level penalization"
python simple_experiment.py -m "repeat_any" -v 40 -e 100

echo
echo "Task: Sequence of Numbers"
echo "with upper level penalization"
python simple_experiment.py -m "sequence" -v 40 -e 100 --penalize_upper_levels
echo "no upper level penalization"
python simple_experiment.py -m "sequence" -v 40 -e 100

echo
echo "Task: Sequence of Repeating Numbers"
echo "with upper level penalization"
python simple_experiment.py -m "sequence_double" -v 40 -e 100 --penalize_upper_levels
echo "no upper level penalization"
python simple_experiment.py -m "sequence_double" -v 40 -e 100

echo
echo "Task: Binary Repeating"
echo "with upper level penalization"
python simple_experiment.py -m "binary_count" -v 10 -e 300 --penalize_upper_levels
echo "no upper level penalization"
python simple_experiment.py -m "binary_count" -v 10 -e 300
