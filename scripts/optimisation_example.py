""" Example script for unconstrained optimisation based Kalman smoother predictions """

import os
import pandas as pd
import sys
from eks.multiview_pca_smoother import ensemble_kalman_smoother_multi_cam
from eks.newton_eks import *
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
from scipy.optimize import *
import argparse
from eks.utils import *
from eks.ensemble_kalman import ensemble
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

def eks_opti_smoother_pupil(
        markers_list, keypoint_names, tracker_name, state_transition_matrix, plot=True):
    """

    Parameters
    ----------
    markers_list : list of pd.DataFrames
        each list element is a dataframe of predictions from one ensemble member
    tracker_name : str
        tracker name for constructing final dataframe

    Returns
    -------
    dict
        markers_df: dataframe containing smoothed markers; same format as input dataframes
        latents_df: dataframe containing 3d latents: pupil diameter and pupil center of mass

    """

    # compute ensemble median
    keys = ['pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y',
            'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y']
    ensemble_preds, ensemble_vars, ensemble_stacks, keypoints_mean_dict, keypoints_var_dict, keypoints_stack_dict = ensemble(
        markers_list, keys)
    
    # --------------------------------------
    # Set parameters
    # --------------------------------------
    # compute center of mass
    pupil_locations = get_pupil_location(keypoints_mean_dict)
    pupil_diameters = get_pupil_diameter(keypoints_mean_dict)
    diameters = []
    for i in range(len(markers_list)):
        keypoints_dict = keypoints_stack_dict[i]
        diameter = get_pupil_diameter(keypoints_dict)
        diameters.append(diameter)

    mean_x_obs = np.mean(pupil_locations[:, 0])
    mean_y_obs = np.mean(pupil_locations[:, 1])
    # make the mean zero
    x_t_obs, y_t_obs = pupil_locations[:, 0] - mean_x_obs, pupil_locations[:, 1] - mean_y_obs

    scaled_ensemble_preds = ensemble_preds.copy()
    scaled_ensemble_stacks = ensemble_stacks.copy()
    # subtract COM means from the ensemble predictions
    for i in range(ensemble_preds.shape[1]):
        if i % 2 == 0:
            scaled_ensemble_preds[:, i] -= mean_x_obs
        else:
            scaled_ensemble_preds[:, i] -= mean_y_obs
    # subtract COM means from all the predictions
    for i in range(ensemble_preds.shape[1]):
        if i % 2 == 0:
            scaled_ensemble_stacks[:, :, i] -= mean_x_obs
        else:
            scaled_ensemble_stacks[:, :, i] -= mean_y_obs
    y = scaled_ensemble_preds
    
    mean_array = np.array([0, mean_x_obs, mean_y_obs])
 
    # --------------------------------------
    # Kalman initial parameters 
    # --------------------------------------
    r = 3
    T = 2000
    mu0 = np.asarray([np.mean(pupil_diameters), 0.0, 0.0])

    # diagonal: var
    S0 = np.asarray([
        [np.var(pupil_diameters), 0.0, 0.0],
        [0.0, np.var(x_t_obs), 0.0],
        [0.0, 0.0, np.var(y_t_obs)]
    ])

    A = np.asarray([
        [0.99, 0, 0],
        [0, 0.99, 0],
        [0, 0, 0.99]
    ])
    B = np.asarray([[0, 1, 0], [-.5, 0, 1], [0, 1, 0], 
                    [.5, 0, 1], [.5, 1, 0], [0, 0, 1], 
                    [-.5, 1, 0],[0, 0, 1]])

    # state covariance matrix
    E = np.asarray([
            [np.var(pupil_diameters) * (1 - (A[0, 0] ** 2)), 0, 0],
            [0, np.var(x_t_obs) * (1 - A[1, 1] ** 2), 0],
            [0, 0, np.var(y_t_obs) * (1 - (A[2, 2] ** 2))]
        ])

    D = np.eye(8)
    # --------------------------------------
    # Run optimisation EKS
    # --------------------------------------
    q_test = kalman_newton_recursive(y, mu0, S0, A, B, ensemble_vars, E)

    if plot:
        # Change file path to your own
        pupil_latent = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/pupil-test/kalman_smoothed_latents.csv", header=[0,1], index_col=0)

    # Latent result from standard eks
        q = pupil_latent.to_numpy()
        latent_plots(q_test,q, mean_array, n=2000)
        
    # Smoothed posterior over y
    y_m_smooth = np.dot(B, q.T).T
    
    # --------------------------------------
    # cleanup
    # --------------------------------------
    # save out marker info
    pdindex = make_dlc_pandas_index(keypoint_names)
    processed_arr_dict = add_mean_to_array(y_m_smooth, keys, mean_x_obs, mean_y_obs)
    key_pair_list = [['pupil_top_r_x', 'pupil_top_r_y'],
                     ['pupil_right_r_x', 'pupil_right_r_y'],
                     ['pupil_bottom_r_x', 'pupil_bottom_r_y'],
                     ['pupil_left_r_x', 'pupil_left_r_y']]
    pred_arr = []
    for key_pair in key_pair_list:
        pred_arr.append(processed_arr_dict[key_pair[0]])
        pred_arr.append(processed_arr_dict[key_pair[1]])
        var = np.empty(processed_arr_dict[key_pair[0]].shape)
        var[:] = np.nan
        pred_arr.append(var)
    pred_arr = np.asarray(pred_arr)
    markers_df = pd.DataFrame(pred_arr.T, columns=pdindex)

    # save out latents info: pupil diam, center of mass
    pred_arr2 = []
    pred_arr2.append(q_test[:, 0])
    pred_arr2.append(q_test[:, 1] + mean_x_obs)  # add back x mean of pupil location
    pred_arr2.append(q_test[:, 2] + mean_y_obs)  # add back y mean of pupil location
    pred_arr2 = np.asarray(pred_arr2)
    arrays = [[tracker_name, tracker_name, tracker_name], ['diameter', 'com_x', 'com_y']]
    pd_index2 = pd.MultiIndex.from_arrays(arrays, names=('scorer', 'latent'))
    latents_df = pd.DataFrame(pred_arr2.T, columns=pd_index2)

    return {'markers_df': markers_df, 'latents_df': latents_df}


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

    
    
    
#%%%%%%%%%%%%%%%%%%% Multi camera example with fish data unconstraint #%%%%%%%%%%%%%%%%%%% 