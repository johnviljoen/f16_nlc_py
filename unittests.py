'''
This script will run through the unit tests for each directory. These
are contained within each directories respective validation file.

This hierarchy of complexity was chosen as each subsequent directory
that is tested builds on the previous one.
'''

import utils.validation
import tables.validation
import dynamics.validation
import control.validation