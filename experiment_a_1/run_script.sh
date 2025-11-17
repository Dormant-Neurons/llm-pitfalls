#!/bin/bash

# The variable ${QUICK_CHECK} is passed from the docker run command

# Call the experiments for the three categories in the paper (adjust as you need)
python3 run_experiment.py --target_category vaccine --quick ${QUICK_CHECK}
python3 run_experiment.py --target_category rus_ukr --quick ${QUICK_CHECK}
python3 run_experiment.py --target_category us_capitol --quick ${QUICK_CHECK}
python3 aggregate_results.py