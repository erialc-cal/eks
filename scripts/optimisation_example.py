""" Example script for unconstrained optimisation based Kalman smoother predictions """

import os
import pandas as pd
import sys
from eks.utils import convert_lp_dlc
from eks.multiview_pca_smoother import ensemble_kalman_smoother_multi_cam
from eks.newton_eks import *
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
from scipy.optimize import *


# Pupil example single camera unconstraint







# Multi camera example with fish data unconstraint