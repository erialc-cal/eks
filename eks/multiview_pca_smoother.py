import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from eks.newton_eks import *
from eks.utils import make_dlc_pandas_index
from eks.ensemble_kalman import ensemble, filtering_pass, kalman_dot, smooth_backward


# -----------------------
# funcs for kalman paw
# -----------------------
def remove_camera_means(ensemble_stacks, camera_means):
    scaled_ensemble_stacks = ensemble_stacks.copy()
    for k in range(len(ensemble_stacks)):
        for camera_id, camera_mean in enumerate(camera_means):
            scaled_ensemble_stacks[k][:,camera_id] = ensemble_stacks[k][:,camera_id] - camera_mean
    return scaled_ensemble_stacks


def add_camera_means(ensemble_stacks, camera_means):
    scaled_ensemble_stacks = ensemble_stacks.copy()
    for k in range(len(ensemble_stacks)):
        for camera_id, camera_mean in enumerate(camera_means):
            scaled_ensemble_stacks[k][:,camera_id] = ensemble_stacks[k][:,camera_id] + camera_mean
    return scaled_ensemble_stacks


def pca(S, n_comps):
    pca_ = PCA(n_components=n_comps)
    return pca_.fit(S), pca_.explained_variance_ratio_


def ensemble_kalman_smoother_paw_asynchronous(
        markers_list_left_cam, markers_list_right_cam, timestamps_left_cam,
        timestamps_right_cam, keypoint_names, smooth_param, quantile_keep_pca):
    """Use multi-view constraints to fit a 3d latent subspace for each body part.

    Parameters
    ----------
    markers_list_left_cam : list of pd.DataFrames
        each list element is a dataframe of predictions from one ensemble member (left view)
    markers_list_right_cam : list of pd.DataFrames
        each list element is a dataframe of predictions from one ensemble member (right view)
    timestamps_left_cam : np.array
        same length as dfs in markers_list_left_cam
    timestamps_right_cam : np.array
        same length as dfs in markers_list_right_cam
    keypoint_names : list
        list of names in the order they appear in marker dfs
    smooth_param : float
        ranges from .01-2 (smaller values = more smoothing)
    quantile_keep_pca
        percentage of the points are kept for multi-view PCA (lowest ensemble variance)

    Returns
    -------

    Returns
    -------
    dict
        markers_df: dataframe containing smoothed markers; same format as input dataframes
        latents_df: dataframe containing 3d latents: multiview PCA

    """

    # --------------------------------------------------------------
    # interpolate right cam markers to left cam timestamps
    # --------------------------------------------------------------
    markers_list_stacked_interp = []
    markers_list_interp = [[], []]
    img_width = 128
    for model_id in range(len(markers_list_left_cam)):
        bl_markers_curr = []
        left_markers_curr = []
        right_markers_curr = []
        bl_left_np = markers_list_left_cam[model_id].to_numpy()
        bl_right_np = markers_list_right_cam[model_id].to_numpy()
        bl_right_interp = []
        for i in range(bl_left_np.shape[1]):
            bl_right_interp.append(interp1d(timestamps_right_cam, bl_right_np[:, i]))
        for i, ts in enumerate(timestamps_left_cam):
            if ts > timestamps_right_cam[-1]:
                break
            if ts < timestamps_right_cam[0]:
                continue
            left_markers = np.array(bl_left_np[i, [0, 1, 3, 4]])
            left_markers_curr.append(left_markers)
            right_markers = np.array([bl_right_interp[j](ts) for j in [0, 1, 3, 4]])
            right_markers[0] = img_width - right_markers[0]  # flip points to match left camera
            right_markers[2] = img_width - right_markers[2]  # flip points to match left camera
            right_markers_curr.append(right_markers)
            # combine paw 1 predictions for both cameras
            bl_markers_curr.append(np.concatenate((left_markers[:2], right_markers[:2])))
            # combine paw 2 predictions for both cameras
            bl_markers_curr.append(np.concatenate((left_markers[2:4], right_markers[2:4])))
        markers_list_stacked_interp.append(bl_markers_curr)
        markers_list_interp[0].append(left_markers_curr)
        markers_list_interp[1].append(right_markers_curr)
    # markers_list_stacked_interp = np.asarray(markers_list_stacked_interp)
    markers_list_interp = np.asarray(markers_list_interp)

    keys = ['paw_l_x', 'paw_l_y', 'paw_r_x', 'paw_r_y']
    markers_list_left_cam = []
    for k in range(len(markers_list_interp[0])):
        markers_left_cam = pd.DataFrame(markers_list_interp[0][k], columns=keys)
        markers_list_left_cam.append(markers_left_cam)

    markers_list_right_cam = []
    for k in range(len(markers_list_interp[1])):
        markers_right_cam = pd.DataFrame(markers_list_interp[1][k], columns=keys)
        markers_list_right_cam.append(markers_right_cam)

    # compute ensemble median left camera
    left_cam_ensemble_preds, left_cam_ensemble_vars, left_cam_ensemble_stacks, left_cam_keypoints_mean_dict, left_cam_keypoints_var_dict, left_cam_keypoints_stack_dict =  ensemble(markers_list_left_cam, keys)

    # compute ensemble median right camera
    right_cam_ensemble_preds, right_cam_ensemble_vars, right_cam_ensemble_stacks, right_cam_keypoints_mean_dict, right_cam_keypoints_var_dict, right_cam_keypoints_stack_dict = ensemble(markers_list_right_cam, keys)

    # ensemble_stacked = np.median(markers_list_stacked_interp, 0)
    # ensemble_stacked_vars = np.var(markers_list_stacked_interp, 0)

    # keep percentage of the points for multi-view PCA based lowest ensemble variance
    hstacked_vars = np.hstack((left_cam_ensemble_vars, right_cam_ensemble_vars))
    max_vars = np.max(hstacked_vars, 1)
    good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep_pca))[0]

    good_left_cam_ensemble_preds = left_cam_ensemble_preds[good_frames]
    good_right_cam_ensemble_preds = right_cam_ensemble_preds[good_frames]
    good_left_cam_ensemble_vars = left_cam_ensemble_vars[good_frames]
    good_right_cam_ensemble_vars = right_cam_ensemble_vars[good_frames]

    # stack left and right camera predictions and variances for both paws on all the good
    # frames
    good_stacked_ensemble_preds = np.zeros(
        (good_frames.shape[0] * 2, left_cam_ensemble_preds.shape[1]))
    good_stacked_ensemble_vars = np.zeros(
        (good_frames.shape[0] * 2, left_cam_ensemble_vars.shape[1]))
    i, j = 0, 0
    while i < good_right_cam_ensemble_preds.shape[0]:
        good_stacked_ensemble_preds[j] = np.concatenate(
            (good_left_cam_ensemble_preds[i][:2], good_right_cam_ensemble_preds[i][:2]))
        good_stacked_ensemble_vars[j] = np.concatenate(
            (good_left_cam_ensemble_vars[i][:2], good_right_cam_ensemble_vars[i][:2]))
        j += 1
        good_stacked_ensemble_preds[j] = np.concatenate(
            (good_left_cam_ensemble_preds[i][2:4], good_right_cam_ensemble_preds[i][2:4]))
        good_stacked_ensemble_vars[j] = np.concatenate(
            (good_left_cam_ensemble_vars[i][2:4], good_right_cam_ensemble_vars[i][2:4]))
        i += 1
        j += 1

    # combine left and right camera predictions and variances for both paws
    left_paw_ensemble_preds = np.zeros(
        (left_cam_ensemble_preds.shape[0], left_cam_ensemble_preds.shape[1]))
    right_paw_ensemble_preds = np.zeros(
        (right_cam_ensemble_preds.shape[0], right_cam_ensemble_preds.shape[1]))
    left_paw_ensemble_vars = np.zeros(
        (left_cam_ensemble_vars.shape[0], left_cam_ensemble_vars.shape[1]))
    right_paw_ensemble_vars = np.zeros(
        (right_cam_ensemble_vars.shape[0], right_cam_ensemble_vars.shape[1]))
    for i in range(len(left_cam_ensemble_preds)):
        left_paw_ensemble_preds[i] = np.concatenate(
            (left_cam_ensemble_preds[i][:2], right_cam_ensemble_preds[i][:2]))
        right_paw_ensemble_preds[i] = np.concatenate(
            (left_cam_ensemble_preds[i][2:4], right_cam_ensemble_preds[i][2:4]))
        left_paw_ensemble_vars[i] = np.concatenate(
            (left_cam_ensemble_vars[i][:2], right_cam_ensemble_vars[i][:2]))
        right_paw_ensemble_vars[i] = np.concatenate(
            (left_cam_ensemble_vars[i][2:4], right_cam_ensemble_vars[i][2:4]))

    # get mean of each paw
    mean_camera_l_x = good_stacked_ensemble_preds[:, 0].mean()
    mean_camera_l_y = good_stacked_ensemble_preds[:, 1].mean()
    mean_camera_r_x = good_stacked_ensemble_preds[:, 2].mean()
    mean_camera_r_y = good_stacked_ensemble_preds[:, 3].mean()
    means_camera = [mean_camera_l_x, mean_camera_l_y, mean_camera_r_x, mean_camera_r_y]

    left_paw_ensemble_stacks = np.concatenate(
        (left_cam_ensemble_stacks[:, :, :2], right_cam_ensemble_stacks[:, :, :2]), 2)
    scaled_left_paw_ensemble_stacks = remove_camera_means(
        left_paw_ensemble_stacks, means_camera)

    right_paw_ensemble_stacks = np.concatenate(
        (right_cam_ensemble_stacks[:, :, :2], right_cam_ensemble_stacks[:, :, :2]), 2)
    scaled_right_paw_ensemble_stacks = remove_camera_means(
        right_paw_ensemble_stacks, means_camera)

    good_scaled_stacked_ensemble_preds = \
        remove_camera_means(good_stacked_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pca, ensemble_ex_var = pca(good_scaled_stacked_ensemble_preds, 3)

    scaled_left_paw_ensemble_preds = \
        remove_camera_means(left_paw_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pcs_left_paw = ensemble_pca.transform(scaled_left_paw_ensemble_preds)
    good_ensemble_pcs_left_paw = ensemble_pcs_left_paw[good_frames]

    scaled_right_paw_ensemble_preds = \
        remove_camera_means(right_paw_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pcs_right_paw = ensemble_pca.transform(scaled_right_paw_ensemble_preds)
    good_ensemble_pcs_right_paw = ensemble_pcs_right_paw[good_frames]

    # --------------------------------------------------------------
    # kalman filtering + smoothing
    # --------------------------------------------------------------
    # $z_t = (d_t, x_t, y_t)$
    # $z_t = A z_{t-1} + e_t, e_t ~ N(0,E)$
    # $O_t = B z_t + n_t, n_t ~ N(0,D_t)$

    dfs = {}
    for paw in ['left', 'right']:

        # --------------------------------------
        # Set values for kalman filter
        # --------------------------------------
        if paw == 'left':
            save_keypoint_name = keypoint_names[0]
            good_ensemble_pcs = good_ensemble_pcs_left_paw
            ensemble_vars = left_paw_ensemble_vars
            y = scaled_left_paw_ensemble_preds
            ensemble_stacks = scaled_left_paw_ensemble_stacks
        else:
            save_keypoint_name = keypoint_names[1]
            good_ensemble_pcs = good_ensemble_pcs_right_paw
            ensemble_vars = right_paw_ensemble_vars
            y = scaled_right_paw_ensemble_preds
            ensemble_stacks = scaled_right_paw_ensemble_stacks

        # compute center of mass
        # latent variables (observed)
        good_z_t_obs = good_ensemble_pcs  # latent variables - true 3D pca

        # Set values for kalman filter #
        # initial state: mean
        m0 = np.asarray([0.0, 0.0, 0.0])

        # diagonal: var
        S0 = np.asarray([
            [np.var(good_z_t_obs[:, 0]), 0.0, 0.0],
            [0.0, np.var(good_z_t_obs[:, 1]), 0.0],
            [0.0, 0.0, np.var(good_z_t_obs[:, 2])]
        ])

        # state-transition matrix
        A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # state covariance matrix
        d_t = good_z_t_obs[1:] - good_z_t_obs[:-1]
        Q = smooth_param * np.cov(d_t.T)

        # measurement function is inverse transform of PCA
        C = ensemble_pca.components_.T

        # placeholder diagonal matrix for ensemble variance
        R = np.eye(ensemble_pca.components_.shape[1])

        # --------------------------------------
        # perform filtering
        # --------------------------------------
        # do filtering pass with time-varying ensemble variances
        print(f"filtering {paw} paw...")
        mf, Vf, S = filtering_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
        print("done filtering")

        # --------------------------------------
        # perform smoothing
        # --------------------------------------
        # Do the smoothing step
        print(f"smoothing {paw} paw...")
        ms, Vs, _ = smooth_backward(y, mf, Vf, S, A, Q, C)
        print("done smoothing")
        # Smoothed posterior over y
        y_m_smooth = np.dot(C, ms.T).T
        y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

        # --------------------------------------
        # cleanup for this paw
        # --------------------------------------
        save_keypoint_names = ['l_cam_' + save_keypoint_name, 'r_cam_' + save_keypoint_name]
        pdindex = make_dlc_pandas_index(save_keypoint_names)

        scaled_y_m_smooth = add_camera_means(y_m_smooth[None, :, :], means_camera)[0]
        pred_arr = []
        for i in range(len(save_keypoint_names)):
            pred_arr.append(scaled_y_m_smooth.T[0 + 2 * i])
            pred_arr.append(scaled_y_m_smooth.T[1 + 2 * i])
            var = np.empty(scaled_y_m_smooth.T[0 + 2 * i].shape)
            var[:] = np.nan
            pred_arr.append(var)
        pred_arr = np.asarray(pred_arr)
        dfs[paw] = pd.DataFrame(pred_arr.T, columns=pdindex)

    # --------------------------------------
    # final cleanup
    # --------------------------------------
    pdindex = make_dlc_pandas_index(keypoint_names)

    # make left cam dataframe
    pred_arr = np.hstack([
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'x')].to_numpy()[:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'y')].to_numpy()[:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'likelihood')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'x')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'y')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'likelihood')].to_numpy()[:, None],
    ])
    df_left = pd.DataFrame(pred_arr, columns=pdindex)

    # make right cam dataframe
    # note we swap left and right paws to match dlc/lp convention
    # note we flip the paws horizontally to match lp convention
    pred_arr = np.hstack([
        img_width - dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'x')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'y')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'likelihood')].to_numpy()[:, None],
        img_width - dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'x')].to_numpy()[:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'y')].to_numpy()[:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'likelihood')].to_numpy()[:, None],
    ])
    df_right = pd.DataFrame(pred_arr, columns=pdindex)

    return {'left_df': df_left, 'right_df': df_right}



def eks_opti_smoother_paw_asynchronous(
        markers_list_left_cam, markers_list_right_cam, timestamps_left_cam,
        timestamps_right_cam, keypoint_names, smooth_param, quantile_keep_pca):
    """Use multi-view constraints to fit a 3d latent subspace for each body part.

    Parameters
    ----------
    markers_list_left_cam : list of pd.DataFrames
        each list element is a dataframe of predictions from one ensemble member (left view)
    markers_list_right_cam : list of pd.DataFrames
        each list element is a dataframe of predictions from one ensemble member (right view)
    timestamps_left_cam : np.array
        same length as dfs in markers_list_left_cam
    timestamps_right_cam : np.array
        same length as dfs in markers_list_right_cam
    keypoint_names : list
        list of names in the order they appear in marker dfs
    smooth_param : float
        ranges from .01-2 (smaller values = more smoothing)
    quantile_keep_pca
        percentage of the points are kept for multi-view PCA (lowest ensemble variance)

    Returns
    -------

    Returns
    -------
    dict
        markers_df: dataframe containing smoothed markers; same format as input dataframes
        latents_df: dataframe containing 3d latents: multiview PCA

    """

    # --------------------------------------------------------------
    # interpolate right cam markers to left cam timestamps
    # --------------------------------------------------------------
    markers_list_stacked_interp = []
    markers_list_interp = [[], []]
    img_width = 128
    for model_id in range(len(markers_list_left_cam)):
        bl_markers_curr = []
        left_markers_curr = []
        right_markers_curr = []
        bl_left_np = markers_list_left_cam[model_id].to_numpy()
        bl_right_np = markers_list_right_cam[model_id].to_numpy()
        bl_right_interp = []
        for i in range(bl_left_np.shape[1]):
            bl_right_interp.append(interp1d(timestamps_right_cam, bl_right_np[:, i]))
        for i, ts in enumerate(timestamps_left_cam):
            if ts > timestamps_right_cam[-1]:
                break
            if ts < timestamps_right_cam[0]:
                continue
            left_markers = np.array(bl_left_np[i, [0, 1, 3, 4]])
            left_markers_curr.append(left_markers)
            right_markers = np.array([bl_right_interp[j](ts) for j in [0, 1, 3, 4]])
            right_markers[0] = img_width - right_markers[0]  # flip points to match left camera
            right_markers[2] = img_width - right_markers[2]  # flip points to match left camera
            right_markers_curr.append(right_markers)
            # combine paw 1 predictions for both cameras
            bl_markers_curr.append(np.concatenate((left_markers[:2], right_markers[:2])))
            # combine paw 2 predictions for both cameras
            bl_markers_curr.append(np.concatenate((left_markers[2:4], right_markers[2:4])))
        markers_list_stacked_interp.append(bl_markers_curr)
        markers_list_interp[0].append(left_markers_curr)
        markers_list_interp[1].append(right_markers_curr)
    # markers_list_stacked_interp = np.asarray(markers_list_stacked_interp)
    markers_list_interp = np.asarray(markers_list_interp)

    keys = ['paw_l_x', 'paw_l_y', 'paw_r_x', 'paw_r_y']
    markers_list_left_cam = []
    for k in range(len(markers_list_interp[0])):
        markers_left_cam = pd.DataFrame(markers_list_interp[0][k], columns=keys)
        markers_list_left_cam.append(markers_left_cam)

    markers_list_right_cam = []
    for k in range(len(markers_list_interp[1])):
        markers_right_cam = pd.DataFrame(markers_list_interp[1][k], columns=keys)
        markers_list_right_cam.append(markers_right_cam)

    # compute ensemble median left camera
    left_cam_ensemble_preds, left_cam_ensemble_vars, left_cam_ensemble_stacks, left_cam_keypoints_mean_dict, left_cam_keypoints_var_dict, left_cam_keypoints_stack_dict =  ensemble(markers_list_left_cam, keys)

    # compute ensemble median right camera
    right_cam_ensemble_preds, right_cam_ensemble_vars, right_cam_ensemble_stacks, right_cam_keypoints_mean_dict, right_cam_keypoints_var_dict, right_cam_keypoints_stack_dict = ensemble(markers_list_right_cam, keys)

    # ensemble_stacked = np.median(markers_list_stacked_interp, 0)
    # ensemble_stacked_vars = np.var(markers_list_stacked_interp, 0)

    # keep percentage of the points for multi-view PCA based lowest ensemble variance
    hstacked_vars = np.hstack((left_cam_ensemble_vars, right_cam_ensemble_vars))
    max_vars = np.max(hstacked_vars, 1)
    good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep_pca))[0]

    good_left_cam_ensemble_preds = left_cam_ensemble_preds[good_frames]
    good_right_cam_ensemble_preds = right_cam_ensemble_preds[good_frames]
    good_left_cam_ensemble_vars = left_cam_ensemble_vars[good_frames]
    good_right_cam_ensemble_vars = right_cam_ensemble_vars[good_frames]

    # stack left and right camera predictions and variances for both paws on all the good
    # frames
    good_stacked_ensemble_preds = np.zeros(
        (good_frames.shape[0] * 2, left_cam_ensemble_preds.shape[1]))
    good_stacked_ensemble_vars = np.zeros(
        (good_frames.shape[0] * 2, left_cam_ensemble_vars.shape[1]))
    i, j = 0, 0
    while i < good_right_cam_ensemble_preds.shape[0]:
        good_stacked_ensemble_preds[j] = np.concatenate(
            (good_left_cam_ensemble_preds[i][:2], good_right_cam_ensemble_preds[i][:2]))
        good_stacked_ensemble_vars[j] = np.concatenate(
            (good_left_cam_ensemble_vars[i][:2], good_right_cam_ensemble_vars[i][:2]))
        j += 1
        good_stacked_ensemble_preds[j] = np.concatenate(
            (good_left_cam_ensemble_preds[i][2:4], good_right_cam_ensemble_preds[i][2:4]))
        good_stacked_ensemble_vars[j] = np.concatenate(
            (good_left_cam_ensemble_vars[i][2:4], good_right_cam_ensemble_vars[i][2:4]))
        i += 1
        j += 1

    # combine left and right camera predictions and variances for both paws
    left_paw_ensemble_preds = np.zeros(
        (left_cam_ensemble_preds.shape[0], left_cam_ensemble_preds.shape[1]))
    right_paw_ensemble_preds = np.zeros(
        (right_cam_ensemble_preds.shape[0], right_cam_ensemble_preds.shape[1]))
    left_paw_ensemble_vars = np.zeros(
        (left_cam_ensemble_vars.shape[0], left_cam_ensemble_vars.shape[1]))
    right_paw_ensemble_vars = np.zeros(
        (right_cam_ensemble_vars.shape[0], right_cam_ensemble_vars.shape[1]))
    for i in range(len(left_cam_ensemble_preds)):
        left_paw_ensemble_preds[i] = np.concatenate(
            (left_cam_ensemble_preds[i][:2], right_cam_ensemble_preds[i][:2]))
        right_paw_ensemble_preds[i] = np.concatenate(
            (left_cam_ensemble_preds[i][2:4], right_cam_ensemble_preds[i][2:4]))
        left_paw_ensemble_vars[i] = np.concatenate(
            (left_cam_ensemble_vars[i][:2], right_cam_ensemble_vars[i][:2]))
        right_paw_ensemble_vars[i] = np.concatenate(
            (left_cam_ensemble_vars[i][2:4], right_cam_ensemble_vars[i][2:4]))

    # get mean of each paw
    mean_camera_l_x = good_stacked_ensemble_preds[:, 0].mean()
    mean_camera_l_y = good_stacked_ensemble_preds[:, 1].mean()
    mean_camera_r_x = good_stacked_ensemble_preds[:, 2].mean()
    mean_camera_r_y = good_stacked_ensemble_preds[:, 3].mean()
    means_camera = [mean_camera_l_x, mean_camera_l_y, mean_camera_r_x, mean_camera_r_y]

    left_paw_ensemble_stacks = np.concatenate(
        (left_cam_ensemble_stacks[:, :, :2], right_cam_ensemble_stacks[:, :, :2]), 2)
    scaled_left_paw_ensemble_stacks = remove_camera_means(
        left_paw_ensemble_stacks, means_camera)

    right_paw_ensemble_stacks = np.concatenate(
        (right_cam_ensemble_stacks[:, :, :2], right_cam_ensemble_stacks[:, :, :2]), 2)
    scaled_right_paw_ensemble_stacks = remove_camera_means(
        right_paw_ensemble_stacks, means_camera)

    good_scaled_stacked_ensemble_preds = \
        remove_camera_means(good_stacked_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pca, ensemble_ex_var = pca(good_scaled_stacked_ensemble_preds, 3)

    scaled_left_paw_ensemble_preds = \
        remove_camera_means(left_paw_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pcs_left_paw = ensemble_pca.transform(scaled_left_paw_ensemble_preds)
    good_ensemble_pcs_left_paw = ensemble_pcs_left_paw[good_frames]

    scaled_right_paw_ensemble_preds = \
        remove_camera_means(right_paw_ensemble_preds[None, :, :], means_camera)[0]
    ensemble_pcs_right_paw = ensemble_pca.transform(scaled_right_paw_ensemble_preds)
    good_ensemble_pcs_right_paw = ensemble_pcs_right_paw[good_frames]

    # --------------------------------------------------------------
    # kalman filtering + smoothing
    # --------------------------------------------------------------
    # $z_t = (d_t, x_t, y_t)$
    # $z_t = A z_{t-1} + e_t, e_t ~ N(0,E)$
    # $O_t = B z_t + n_t, n_t ~ N(0,D_t)$

    dfs = {}
    for paw in ['left', 'right']:

        # --------------------------------------
        # Set values for kalman filter
        # --------------------------------------
        if paw == 'left':
            save_keypoint_name = keypoint_names[0]
            good_ensemble_pcs = good_ensemble_pcs_left_paw
            ensemble_vars = left_paw_ensemble_vars
            y = scaled_left_paw_ensemble_preds
            ensemble_stacks = scaled_left_paw_ensemble_stacks
        else:
            save_keypoint_name = keypoint_names[1]
            good_ensemble_pcs = good_ensemble_pcs_right_paw
            ensemble_vars = right_paw_ensemble_vars
            y = scaled_right_paw_ensemble_preds
            ensemble_stacks = scaled_right_paw_ensemble_stacks

        # compute center of mass
        # latent variables (observed)
        good_z_t_obs = good_ensemble_pcs  # latent variables - true 3D pca

        # Set values for kalman filter #
        # initial state: mean
        m0 = np.asarray([0.0, 0.0, 0.0])

        # diagonal: var
        S0 = np.asarray([
            [np.var(good_z_t_obs[:, 0]), 0.0, 0.0],
            [0.0, np.var(good_z_t_obs[:, 1]), 0.0],
            [0.0, 0.0, np.var(good_z_t_obs[:, 2])]
        ])

        # state-transition matrix
        A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # state covariance matrix
        d_t = good_z_t_obs[1:] - good_z_t_obs[:-1]
        E = smooth_param * np.cov(d_t.T)

        # measurement function is inverse transform of PCA
        B = ensemble_pca.components_.T

        # placeholder diagonal matrix for ensemble variance
        D = np.eye(ensemble_pca.components_.shape[1])
    
        # --------------------------------------
        # Run EKS opti
        # --------------------------------------
        ms = kalman_newton_recursive(y, m0, S0, A, B, ensemble_vars, E, max_iter = 1)
        #plt.plot(loss)
        # Smoothed posterior over y
        y_m_smooth = np.dot(B, ms.T).T


        # --------------------------------------
        # cleanup for this paw
        # --------------------------------------
        save_keypoint_names = ['l_cam_' + save_keypoint_name, 'r_cam_' + save_keypoint_name]
        pdindex = make_dlc_pandas_index(save_keypoint_names)

        scaled_y_m_smooth = add_camera_means(y_m_smooth[None, :, :], means_camera)[0]
        pred_arr = []
        for i in range(len(save_keypoint_names)):
            pred_arr.append(scaled_y_m_smooth.T[0 + 2 * i])
            pred_arr.append(scaled_y_m_smooth.T[1 + 2 * i])
            var = np.empty(scaled_y_m_smooth.T[0 + 2 * i].shape)
            var[:] = np.nan
            pred_arr.append(var)
        pred_arr = np.asarray(pred_arr)
        dfs[paw] = pd.DataFrame(pred_arr.T, columns=pdindex)

    # --------------------------------------
    # final cleanup
    # --------------------------------------
    pdindex = make_dlc_pandas_index(keypoint_names)

    # make left cam dataframe
    pred_arr = np.hstack([
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'x')].to_numpy()[:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'y')].to_numpy()[:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_l', 'likelihood')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'x')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'y')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'l_cam_paw_r', 'likelihood')].to_numpy()[:, None],
    ])
    df_left = pd.DataFrame(pred_arr, columns=pdindex)

    # make right cam dataframe
    # note we swap left and right paws to match dlc/lp convention
    # note we flip the paws horizontally to match lp convention
    pred_arr = np.hstack([
        img_width - dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'x')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'y')].to_numpy()[:, None],
        dfs['right'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_r', 'likelihood')].to_numpy()[:, None],
        img_width - dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'x')].to_numpy()[:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'y')].to_numpy()[:, None],
        dfs['left'].loc[:, ('ensemble-kalman_tracker', 'r_cam_paw_l', 'likelihood')].to_numpy()[:, None],
    ])
    df_right = pd.DataFrame(pred_arr, columns=pdindex)

    return {'left_df': df_left, 'right_df': df_right}



# -----------------------
# funcs for mirror-mouse
# -----------------------
def ensemble_kalman_smoother_multi_cam(
        markers_list_cameras, keypoint_ensemble, smooth_param, quantile_keep_pca, camera_names):
    """Use multi-view constraints to fit a 3d latent subspace for each body part.

    Parameters
    ----------
    markers_list_cameras : list of list of pd.DataFrames
        each list element is a list of dataframe predictions from one ensemble member for each camera.
    keypoint_ensemble : str
        the name of the keypoint to be ensembled and smoothed
    smooth_param : float
        ranges from .01-2 (smaller values = more smoothing)
    quantile_keep_pca
        percentage of the points are kept for multi-view PCA (lowest ensemble variance)
    camera_names: list
        the camera names (should be the same length as markers_list_cameras).

    Returns
    -------

    Returns
    -------
    dict
        markers_df: dataframe containing smoothed markers; same format as input dataframes
        latents_df: dataframe containing 3d latents: multiview PCA

    """

    # --------------------------------------------------------------
    # interpolate right cam markers to left cam timestamps
    # --------------------------------------------------------------
    num_cameras = len(camera_names)
    markers_list_stacked_interp = []
    markers_list_interp = [[] for i in range(num_cameras)]
    for model_id in range(len(markers_list_cameras[0])):
        bl_markers_curr = []
        camera_markers_curr = [[] for i in range(num_cameras)]
        for i in range(markers_list_cameras[0][0].shape[0]):
            curr_markers = []
            for camera in range(num_cameras):
                markers = np.array(markers_list_cameras[camera][model_id].to_numpy()[i, [0, 1]])
                camera_markers_curr[camera].append(markers)
                curr_markers.append(markers)
            bl_markers_curr.append(np.concatenate(curr_markers)) #combine predictions for both cameras
        markers_list_stacked_interp.append(bl_markers_curr)
        for camera in range(num_cameras):
            markers_list_interp[camera].append(camera_markers_curr[camera])
    markers_list_stacked_interp = np.asarray(markers_list_stacked_interp)
    markers_list_interp = np.asarray(markers_list_interp)
    
    keys = [keypoint_ensemble+'_x', keypoint_ensemble+'_y']
    markers_list_cams = [[] for i in range(num_cameras)]
    for k in range(len(markers_list_interp[0])):
        for camera in range(num_cameras):
            markers_cam = pd.DataFrame(markers_list_interp[camera][k], columns = keys)
            markers_list_cams[camera].append(markers_cam)
            
    #compute ensemble median for each camera
    cam_ensemble_preds = []
    cam_ensemble_vars = []
    cam_ensemble_stacks = []
    cam_keypoints_mean_dict = []
    cam_keypoints_var_dict = []
    cam_keypoints_stack_dict = []
    for camera in range(num_cameras):
        cam_ensemble_preds_curr, cam_ensemble_vars_curr, cam_ensemble_stacks_curr, cam_keypoints_mean_dict_curr, cam_keypoints_var_dict_curr, cam_keypoints_stack_dict_curr = ensemble(markers_list_cams[camera], keys)
        cam_ensemble_preds.append(cam_ensemble_preds_curr)
        cam_ensemble_vars.append(cam_ensemble_vars_curr)
        cam_ensemble_stacks.append(cam_ensemble_stacks_curr)
        cam_keypoints_mean_dict.append(cam_keypoints_mean_dict_curr)
        cam_keypoints_var_dict.append(cam_keypoints_var_dict_curr)
        cam_keypoints_stack_dict.append(cam_keypoints_stack_dict_curr)
        
    #filter by low ensemble variances
    hstacked_vars = np.hstack(cam_ensemble_vars)
    max_vars = np.max(hstacked_vars,1)
    quantile_keep = quantile_keep_pca
    good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep))[0]

    good_cam_ensemble_preds = []
    good_cam_ensemble_vars = []
    for camera in range(num_cameras):
        good_cam_ensemble_preds.append(cam_ensemble_preds[camera][good_frames])
        good_cam_ensemble_vars.append(cam_ensemble_vars[camera][good_frames])

    good_ensemble_preds = np.hstack(good_cam_ensemble_preds)
    good_ensemble_vars = np.hstack(good_cam_ensemble_vars)
    means_camera = []
    for i in range(good_ensemble_preds.shape[1]):
        means_camera.append(good_ensemble_preds[:,i].mean())

    ensemble_preds = np.hstack(cam_ensemble_preds)
    ensemble_vars = np.hstack(cam_ensemble_vars)
    ensemble_stacks = np.concatenate(cam_ensemble_stacks,2)
    scaled_ensemble_stacks = remove_camera_means(ensemble_stacks, means_camera)

    good_scaled_ensemble_preds = remove_camera_means(good_ensemble_preds[None,:,:], means_camera)[0]
    ensemble_pca, ensemble_ex_var = pca(good_scaled_ensemble_preds, 3)

    scaled_ensemble_preds = remove_camera_means(ensemble_preds[None,:,:], means_camera)[0]
    ensemble_pcs = ensemble_pca.transform(scaled_ensemble_preds)
    good_ensemble_pcs = ensemble_pcs[good_frames]

    y_obs = scaled_ensemble_preds
    
    #compute center of mass
    #latent variables (observed)
    good_z_t_obs = good_ensemble_pcs #latent variables - true 3D pca

    ##### Set values for kalman filter #####
    m0 = np.asarray([0.0, 0.0, 0.0]) # initial state: mean
    S0 =  np.asarray([[np.var(good_z_t_obs[:,0]), 0.0, 0.0], [0.0, np.var(good_z_t_obs[:,1]), 0.0], [0.0, 0.0, np.var(good_z_t_obs[:,2])]]) # diagonal: var

    A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) #state-transition matrix,
    # Q = np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]) #state covariance matrix?????
    d_t = good_z_t_obs[1:] - good_z_t_obs[:-1]

    Q = smooth_param*np.cov(d_t.T)

    C = ensemble_pca.components_.T # Measurement function is inverse transform of PCA
    R = np.eye(ensemble_pca.components_.shape[1]) # placeholder diagonal matrix for ensemble variance

    print(f"filtering {keypoint_ensemble}...")
    mf, Vf, S = filtering_pass(y_obs, m0, S0, C, R, A, Q, ensemble_vars)
    print("done filtering")
    y_m_filt = np.dot(C, mf.T).T
    y_v_filt = np.swapaxes(np.dot(C, np.dot(Vf, C.T)), 0, 1)

    # Do the smoothing step
    print(f"smoothing {keypoint_ensemble}...")
    ms, Vs, _ = smooth_backward(y_obs, mf, Vf, S, A, Q, C)
    print("done smoothing")

    # Smoothed posterior over y
    y_m_smooth = np.dot(C, ms.T).T
    y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

    # --------------------------------------
    # final cleanup
    # --------------------------------------
    pdindex = make_dlc_pandas_index([keypoint_ensemble])
    camera_indices = []
    for camera in range(num_cameras):
        camera_indices.append([camera*2, camera*2+1])
 
    camera_dfs = {}
    for camera, camera_name in enumerate(camera_names):
        var = np.empty(y_m_smooth.T[camera_indices[camera][0]].shape)
        var[:] = np.nan
        pred_arr = np.vstack([
            y_m_smooth.T[camera_indices[camera][0]] + means_camera[camera_indices[camera][0]],
            y_m_smooth.T[camera_indices[camera][1]] + means_camera[camera_indices[camera][1]],
            var,
        ]).T
        camera_dfs[camera_name + '_df'] = pd.DataFrame(pred_arr, columns=pdindex)

    return camera_dfs






# -----------------------
# funcs for mirror-mouse
# -----------------------
def eks_opti_smoother_multi_cam(
        markers_list_cameras, keypoint_ensemble, smooth_param, quantile_keep_pca, camera_names, plot=True):
    """Use multi-view constraints to fit a 3d latent subspace for each body part.

    Parameters
    ----------
    markers_list_cameras : list of list of pd.DataFrames
        each list element is a list of dataframe predictions from one ensemble member for each camera.
    keypoint_ensemble : str
        the name of the keypoint to be ensembled and smoothed
    smooth_param : float
        ranges from .01-2 (smaller values = more smoothing)
    quantile_keep_pca
        percentage of the points are kept for multi-view PCA (lowest ensemble variance)
    camera_names: list
        the camera names (should be the same length as markers_list_cameras).

    Returns
    -------

    Returns
    -------
    dict
        markers_df: dataframe containing smoothed markers; same format as input dataframes
        latents_df: dataframe containing 3d latents: multiview PCA

    """

    # --------------------------------------------------------------
    # interpolate right cam markers to left cam timestamps
    # --------------------------------------------------------------
    num_cameras = len(camera_names)
    markers_list_stacked_interp = []
    markers_list_interp = [[] for i in range(num_cameras)]
    for model_id in range(len(markers_list_cameras[0])):
        bl_markers_curr = []
        camera_markers_curr = [[] for i in range(num_cameras)]
        for i in range(markers_list_cameras[0][0].shape[0]):
            curr_markers = []
            for camera in range(num_cameras):
                markers = np.array(markers_list_cameras[camera][model_id].to_numpy()[i, [0, 1]])
                camera_markers_curr[camera].append(markers)
                curr_markers.append(markers)
            bl_markers_curr.append(np.concatenate(curr_markers)) #combine predictions for both cameras
        markers_list_stacked_interp.append(bl_markers_curr)
        for camera in range(num_cameras):
            markers_list_interp[camera].append(camera_markers_curr[camera])
    markers_list_stacked_interp = np.asarray(markers_list_stacked_interp)
    markers_list_interp = np.asarray(markers_list_interp)
    
    keys = [keypoint_ensemble+'_x', keypoint_ensemble+'_y']
    markers_list_cams = [[] for i in range(num_cameras)]
    for k in range(len(markers_list_interp[0])):
        for camera in range(num_cameras):
            markers_cam = pd.DataFrame(markers_list_interp[camera][k], columns = keys)
            markers_list_cams[camera].append(markers_cam)
            
    #compute ensemble median for each camera
    cam_ensemble_preds = []
    cam_ensemble_vars = []
    cam_ensemble_stacks = []
    cam_keypoints_mean_dict = []
    cam_keypoints_var_dict = []
    cam_keypoints_stack_dict = []
    for camera in range(num_cameras):
        cam_ensemble_preds_curr, cam_ensemble_vars_curr, cam_ensemble_stacks_curr, cam_keypoints_mean_dict_curr, cam_keypoints_var_dict_curr, cam_keypoints_stack_dict_curr = ensemble(markers_list_cams[camera], keys)
        cam_ensemble_preds.append(cam_ensemble_preds_curr)
        cam_ensemble_vars.append(cam_ensemble_vars_curr)
        cam_ensemble_stacks.append(cam_ensemble_stacks_curr)
        cam_keypoints_mean_dict.append(cam_keypoints_mean_dict_curr)
        cam_keypoints_var_dict.append(cam_keypoints_var_dict_curr)
        cam_keypoints_stack_dict.append(cam_keypoints_stack_dict_curr)
        
    #filter by low ensemble variances
    hstacked_vars = np.hstack(cam_ensemble_vars)
    max_vars = np.max(hstacked_vars,1)
    quantile_keep = quantile_keep_pca
    good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep))[0]

    good_cam_ensemble_preds = []
    good_cam_ensemble_vars = []
    for camera in range(num_cameras):
        good_cam_ensemble_preds.append(cam_ensemble_preds[camera][good_frames])
        good_cam_ensemble_vars.append(cam_ensemble_vars[camera][good_frames])

    good_ensemble_preds = np.hstack(good_cam_ensemble_preds)
    good_ensemble_vars = np.hstack(good_cam_ensemble_vars)
    means_camera = []
    for i in range(good_ensemble_preds.shape[1]):
        means_camera.append(good_ensemble_preds[:,i].mean())

    ensemble_preds = np.hstack(cam_ensemble_preds)
    ensemble_vars = np.hstack(cam_ensemble_vars)
    ensemble_stacks = np.concatenate(cam_ensemble_stacks,2)
    scaled_ensemble_stacks = remove_camera_means(ensemble_stacks, means_camera)

    good_scaled_ensemble_preds = remove_camera_means(good_ensemble_preds[None,:,:], means_camera)[0]
    ensemble_pca, ensemble_ex_var = pca(good_scaled_ensemble_preds, 3)

    scaled_ensemble_preds = remove_camera_means(ensemble_preds[None,:,:], means_camera)[0]
    ensemble_pcs = ensemble_pca.transform(scaled_ensemble_preds)
    good_ensemble_pcs = ensemble_pcs[good_frames]

    y_obs = scaled_ensemble_preds
    
    #compute center of mass
    #latent variables (observed)
    good_z_t_obs = good_ensemble_pcs #latent variables - true 3D pca

    ##### Set values for kalman filter #####
    m0 = np.asarray([0.0, 0.0, 0.0]) # initial state: mean
    S0 =  np.asarray([[np.var(good_z_t_obs[:,0]), 0.0, 0.0], [0.0, np.var(good_z_t_obs[:,1]), 0.0], [0.0, 0.0, np.var(good_z_t_obs[:,2])]]) # diagonal: var

    A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) #state-transition matrix,
    # Q = np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]) #state covariance matrix?????
    d_t = good_z_t_obs[1:] - good_z_t_obs[:-1]

    E = smooth_param*np.cov(d_t.T)

    B = ensemble_pca.components_.T # Measurement function is inverse transform of PCA
    D = np.eye(ensemble_pca.components_.shape[1]) # placeholder diagonal matrix for ensemble variance
    
    # --------------------------------------
    # Run EKS opti
    # --------------------------------------
    ms = kalman_newton_recursive(y_obs, m0, S0, A, B, ensemble_vars, E, max_iter = 1)
    #plt.plot(loss)
    # Smoothed posterior over y
    y_m_smooth = np.dot(B, ms.T).T

    if plot:
        # Change file path to your own
        truth = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/pupil-test/kalman_smoothed_latents.csv", header=[0,1], index_col=0)

    # Latent result from standard eks
        q = pupil_latent.to_numpy()
        latent_plots(q_test,q, mean_array, n=2000)
    # --------------------------------------
    # final cleanup
    # --------------------------------------
    pdindex = make_dlc_pandas_index([keypoint_ensemble])
    camera_indices = []
    for camera in range(num_cameras):
        camera_indices.append([camera*2, camera*2+1])
 
    camera_dfs = {}
    for camera, camera_name in enumerate(camera_names):
        var = np.empty(y_m_smooth.T[camera_indices[camera][0]].shape)
        var[:] = np.nan
        pred_arr = np.vstack([
            y_m_smooth.T[camera_indices[camera][0]] + means_camera[camera_indices[camera][0]],
            y_m_smooth.T[camera_indices[camera][1]] + means_camera[camera_indices[camera][1]],
            var,
        ]).T
        camera_dfs[camera_name + '_df'] = pd.DataFrame(pred_arr, columns=pdindex)

    return camera_dfs
