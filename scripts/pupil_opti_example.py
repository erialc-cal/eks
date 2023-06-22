""" Example script for unconstrained optimisation based Kalman smoother predictions """

import os
import pandas as pd
import sys
from eks.newton_eks import *
from scipy.optimize import *
import argparse
from eks.utils import *
from eks.pupil_smoother import eks_opti_smoother_pupil
from eks.pupil_utils import *


#%%%%%%%%%%%%%%%%%%% Pupil example single camera unconstraint #%%%%%%%%%%%%%%%%%%%



# --------------------------------------
# Preprocess data
# --------------------------------------

# Change file path to your own
pupil1 = "/Users/clairehe/Documents/GitHub/eks/data/ibl-pupil/5285c561-80da-4563-8694-739da92e5dd0.left.rng=0.csv"
df_pupil = pd.read_csv(pupil1, header=[0,1,2], index_col=0)
df_pupil.head()

# %run -i 'scripts/pupil_example.py' --csv-dir 'data/ibl-pupil' --save-dir 'data/misc/pupil-test/' --diameter-s 0.99 --com-s 0.99 

from eks.utils import make_dlc_pandas_index
from eks.ensemble_kalman import ensemble
from eks.pupil_utils import get_pupil_location, get_pupil_diameter


parser = argparse.ArgumentParser()
parser.add_argument(
    '--csv-dir',
    required=True,
    help='directory of models for ensembling',
    type=str
)
parser.add_argument(
    '--save-dir',
    help='save directory for outputs (default is csv-dir)',
    default=None,
    type=str,
)
parser.add_argument(
    '--diameter-s',
    help='smoothing parameter for diameter (closer to 1 = more smoothing)',
    default=.9999,
    type=float
)
parser.add_argument(
    '--com-s',
    help='smoothing parameter for center of mass (closer to 1 = more smoothing)',
    default=.999,
    type=float
)
args = parser.parse_args()

# collect user-provided args
csv_dir = os.path.abspath(args.csv_dir)
save_dir = args.save_dir


# ---------------------------------------------
# run EKS algorithm
# ---------------------------------------------

# handle I/O
if not os.path.isdir(csv_dir):
    raise ValueError('--csv-dir must be a valid path to a directory')
    
if save_dir is None:
    save_dir = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(save_dir, exist_ok=True)

# load files and put them in correct format
csv_files = os.listdir(csv_dir)
markers_list = []
for csv_file in csv_files:
    if not csv_file.endswith('csv'):
        continue
    markers_curr = pd.read_csv(os.path.join(csv_dir, csv_file), header=[0, 1, 2], index_col=0)
    keypoint_names = [c[1] for c in markers_curr.columns[::3]]
    model_name = markers_curr.columns[0][0]
    markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
    markers_list.append(markers_curr_fmt)
if len(markers_list) == 0:
    raise FileNotFoundError(f'No marker csv files found in {csv_dir}')

# parameters hand-picked for smoothing purposes (diameter_s, com_s, com_s)
state_transition_matrix = np.asarray([
    [args.diameter_s, 0, 0],
    [0, args.com_s, 0],
    [0, 0, args.com_s]
])
print(f'Smoothing matrix: {state_transition_matrix}')

df_dicts = eks_opti_smoother_pupil(
    markers_list=markers_list,
    keypoint_names=keypoint_names,
    tracker_name='eks-opti_tracker',
    state_transition_matrix=state_transition_matrix,
)

save_file = os.path.join(save_dir, 'opti_eks_pupil_traces.csv')
print(f'saving smoothed predictions to {save_file }')
df_dicts['markers_df'].to_csv(save_file)

save_file = os.path.join(save_dir, 'opti_eks_latents.csv')
print(f'saving latents to {save_file}')
df_dicts['latents_df'].to_csv(save_file)

    