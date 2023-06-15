import pandas as pd
import numpy as np
import warnings


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
