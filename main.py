# main.py

import os
import sys
import numpy as np
from src.evaluation.evaluation_runner import run_all_evaluations



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

if __name__ == '__main__':
    # Set the flag to control plotting (True for showing plots, False to disable)
    show_plots = False
    # Run all evaluations with the configurations provided
    run_all_evaluations(show_plots=show_plots, max_time=1000)
