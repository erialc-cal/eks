import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from eks.newton_eks import *
from sklearn.decomposition import PCA
from eks.utils import make_dlc_pandas_index
from eks.ensemble_kalman import ensemble, filtering_pass, kalman_dot, smooth_backward
import warnings


# -----------------------
# funcs for kalman pupil
# -----------------------
def get_pupil_location(dlc):
    """get mean of both pupil diameters
    d1 = top - bottom, d2 = left - right
    and in addition assume it's a circle and
    estimate diameter from other pairs of points
    Author: Michael Schartner
    """
    s = 1
    t = np.vstack((dlc['pupil_top_r_x'], dlc['pupil_top_r_y'])).T / s
    b = np.vstack((dlc['pupil_bottom_r_x'], dlc['pupil_bottom_r_y'])).T / s
    l = np.vstack((dlc['pupil_left_r_x'], dlc['pupil_left_r_y'])).T / s
    r = np.vstack((dlc['pupil_right_r_x'], dlc['pupil_right_r_y'])).T / s
    center = np.zeros(t.shape)

    # ok if either top or bottom is nan in x-dir
    tmp_x1 = np.nanmedian(np.hstack([t[:, 0, None], b[:, 0, None]]), axis=1)
    # both left and right must be present in x-dir
    tmp_x2 = np.median(np.hstack([r[:, 0, None], l[:, 0, None]]), axis=1)
    center[:, 0] = np.nanmedian(np.hstack([tmp_x1[:, None], tmp_x2[:, None]]), axis=1)

    # both top and bottom must be present in y-dir
    tmp_y1 = np.median(np.hstack([t[:, 1, None], b[:, 1, None]]), axis=1)
    # ok if either left or right is nan in y-dir
    tmp_y2 = np.nanmedian(np.hstack([r[:, 1, None], l[:, 1, None]]), axis=1)
    center[:, 1] = np.nanmedian(np.hstack([tmp_y1[:, None], tmp_y2[:, None]]), axis=1)
    return center


def get_pupil_diameter(dlc):
    """
    from: https://int-brain-lab.github.io/iblenv/_modules/brainbox/behavior/dlc.html
    Estimates pupil diameter by taking median of different computations.

    The two most straightforward estimates: d1 = top - bottom, d2 = left - right
    In addition, assume the pupil is a circle and estimate diameter from other pairs of points

    :param dlc: dlc pqt table with pupil estimates, should be likelihood thresholded (e.g. at 0.9)
    :return: np.array, pupil diameter estimate for each time point, shape (n_frames,)
    """
    diameters = []
    # Get the x,y coordinates of the four pupil points
    top, bottom, left, right = [np.vstack((dlc[f'pupil_{point}_r_x'], dlc[f'pupil_{point}_r_y']))
                                for point in ['top', 'bottom', 'left', 'right']]
    # First compute direct diameters
    diameters.append(np.linalg.norm(top - bottom, axis=0))
    diameters.append(np.linalg.norm(left - right, axis=0))

    # For non-crossing edges, estimate diameter via circle assumption
    for pair in [(top, left), (top, right), (bottom, left), (bottom, right)]:
        diameters.append(np.linalg.norm(pair[0] - pair[1], axis=0) * 2 ** 0.5)

    # Ignore all nan runtime warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(diameters, axis=0)


def add_mean_to_array(pred_arr, keys, mean_x, mean_y):
    pred_arr_copy = pred_arr.copy()
    processed_arr_dict = {}
    for i, key in enumerate(keys):
        if 'x' in key:
            processed_arr_dict[key] = pred_arr_copy[:, i] + mean_x
        else:
            processed_arr_dict[key] = pred_arr_copy[:, i] + mean_y
    return processed_arr_dict


def ensemble_kalman_smoother_pupil(
        markers_list, keypoint_names, tracker_name, state_transition_matrix):
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

    # ## Set parameters
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

    # latent variables (observed)
    # latent variables - diameter, com_x, com_y
    # z_t_obs = np.vstack((pupil_diameters, x_t_obs, y_t_obs))

    # --------------------------------------
    # Set values for kalman filter
    # --------------------------------------
    # initial state: mean
    m0 = np.asarray([np.mean(pupil_diameters), 0.0, 0.0])

    # diagonal: var
    S0 = np.asarray([
        [np.var(pupil_diameters), 0.0, 0.0],
        [0.0, np.var(x_t_obs), 0.0],
        [0.0, 0.0, np.var(y_t_obs)]
    ])

    # state-transition matrix
    A = state_transition_matrix

    # state covariance matrix
    Q = np.asarray([
        [np.var(pupil_diameters) * (1 - (A[0, 0] ** 2)), 0, 0],
        [0, np.var(x_t_obs) * (1 - A[1, 1] ** 2), 0],
        [0, 0, np.var(y_t_obs) * (1 - (A[2, 2] ** 2))]
    ])

    # Measurement function
    C = np.asarray(
        [[0, 1, 0], [-.5, 0, 1], [0, 1,
                                  0], [.5, 0, 1], [.5, 1, 0], [0, 0, 1], [-.5, 1, 0],
         [0, 0, 1]])

    # placeholder diagonal matrix for ensemble variance
    R = np.eye(8)

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

    # --------------------------------------
    # perform filtering
    # --------------------------------------
    # do filtering pass with time-varying ensemble variances
    print("filtering...")
    mf, Vf, S = filtering_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
    print("done filtering")

    # --------------------------------------
    # perform smoothing
    # --------------------------------------
    # Do the smoothing step
    print("smoothing...")
    ms, Vs, _ = smooth_backward(y, mf, Vf, S, A, Q, C)
    print("done smoothing")
    # Smoothed posterior over y
    y_m_smooth = np.dot(C, ms.T).T
    y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

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
    pred_arr2.append(ms[:, 0])
    pred_arr2.append(ms[:, 1] + mean_x_obs)  # add back x mean of pupil location
    pred_arr2.append(ms[:, 2] + mean_y_obs)  # add back y mean of pupil location
    pred_arr2 = np.asarray(pred_arr2)
    arrays = [[tracker_name, tracker_name, tracker_name], ['diameter', 'com_x', 'com_y']]
    pd_index2 = pd.MultiIndex.from_arrays(arrays, names=('scorer', 'latent'))
    latents_df = pd.DataFrame(pred_arr2.T, columns=pd_index2)

    return {'markers_df': markers_df, 'latents_df': latents_df}



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
