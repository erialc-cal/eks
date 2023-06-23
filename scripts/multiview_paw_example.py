"""Example script for ibl-paw dataset."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from eks.utils import convert_lp_dlc
from eks.multiview_pca_smoother import ensemble_kalman_smoother_paw_asynchronous


parser = argparse.ArgumentParser()
parser.add_argument(
    '--csv-dir',
    required=True,
    help='directory of models for ensembling',
    type=str,
)
parser.add_argument(
    '--save-dir',
    help='save directory for outputs (default is csv-dir)',
    default=None,
    type=str,
)
parser.add_argument(
    '--s',
    help='smoothing parameter ranges from .01-2 (smaller values = more smoothing)',
    default=1,
    type=float,
)
parser.add_argument(
    '--quantile_keep_pca',
    help='percentage of the points are kept for multi-view PCA (lowest ensemble variance)',
    default=25,
    type=float,
)
parser.add_argument(
    '--eks_version',
    required=True,
    help='choose eks version: optimisation based or standard em',
    type=str,
)
args = parser.parse_args()

# collect user-provided args
csv_dir = os.path.abspath(args.csv_dir)
save_dir = args.save_dir
s = args.s
quantile_keep_pca = args.quantile_keep_pca
eks_version = args.eks_version


# ---------------------------------------------
# run EKS algorithm
# ---------------------------------------------

# handle I/O
if not os.path.isdir(csv_dir):
    raise ValueError('csv-dir must be a valid path to a directory')

if save_dir is None:
    save_dir = os.path.join(os.getcwd(), 'outputs')
    os.makedirs(save_dir, exist_ok=True)

# load files and put them in correct format
markers_list_left = []
markers_list_right = []
timestamps_left = None
timestamps_right = None
filenames = os.listdir(csv_dir)
for filename in filenames:
    if 'timestamps' not in filename:
        markers_curr = pd.read_csv(os.path.join(csv_dir, filename), header=[0, 1, 2], index_col=0)
        keypoint_names = [c[1] for c in markers_curr.columns[::3]]
        model_name = markers_curr.columns[0][0]
        markers_curr_fmt = convert_lp_dlc(markers_curr, keypoint_names, model_name=model_name)
        if 'left' in filename:
            markers_list_left.append(markers_curr_fmt)
        else:
            # switch right camera paws
            columns = {
                'paw_l_x': 'paw_r_x', 'paw_l_y': 'paw_r_y',
                'paw_l_likelihood': 'paw_r_likelihood',
                'paw_r_x': 'paw_l_x', 'paw_r_y': 'paw_l_y',
                'paw_r_likelihood': 'paw_l_likelihood'
            }
            markers_curr_fmt = markers_curr_fmt.rename(columns=columns)
            # reorder columns
            markers_curr_fmt = markers_curr_fmt.loc[:, columns.keys()]
            markers_list_right.append(markers_curr_fmt)
    else:
        if 'left' in filename:
            timestamps_left = np.load(os.path.join(csv_dir, filename))
        else:
            timestamps_right = np.load(os.path.join(csv_dir, filename))

# file checks
if timestamps_left is None or timestamps_right is None:
    raise ValueError('Need timestamps for both cameras')
    
if len(markers_list_right) != len(markers_list_left) or len(markers_list_left) == 0:
    raise ValueError(
        'There must be the same number of left and right camera models and >=1 model for each.')

# run eks
    # run eks
if eks_version == "opti":
    df_dicts = eks_opti_smoother_paw_asynchronous(
        markers_list_left_cam=markers_list_left,
        markers_list_right_cam=markers_list_right,
        timestamps_left_cam=timestamps_left,
        timestamps_right_cam=timestamps_right,
        keypoint_names=keypoint_names,
        smooth_param=s,
        quantile_keep_pca=quantile_keep_pca,
    )
else:
    df_dicts = ensemble_kalman_smoother_paw_asynchronous(
    markers_list_left_cam=markers_list_left,
    markers_list_right_cam=markers_list_right,
    timestamps_left_cam=timestamps_left,
    timestamps_right_cam=timestamps_right,
    keypoint_names=keypoint_names,
    smooth_param=s,
    quantile_keep_pca=quantile_keep_pca,
)



# save smoothed markers from each view
for view in ['left', 'right']:
    save_file = os.path.join(save_dir, f'kalman_smoothed_paw_traces.{view}.csv')
    df_dicts[f'{view}_df'].to_csv(save_file)


# ---------------------------------------------
# plot results
# ---------------------------------------------

# select example keypoint from example camera view
kp = keypoint_names[0]
view = 'left'  # NOTE: if you want to use right view, must swap paw identities
idxs = (0, 500)

fig, axes = plt.subplots(3, 1, figsize=(9, 6))

for ax, coord in zip(axes, ['x', 'y', 'likelihood']):
    # plot individual models
    for m, markers_curr in enumerate(markers_list_left):
        ax.plot(
            markers_curr.loc[slice(*idxs), f'{kp}_{coord}'], color=[0.5, 0.5, 0.5],
            label='Individual models' if m == 0 else None,
        )
    ax.set_ylabel(coord, fontsize=12)
    ax.set_xlabel('Time (frames)', fontsize=12)
    # plot eks
    if coord == 'likelihood':
        continue
    ax.plot(
        df_dicts[f'{view}_df'].loc[slice(*idxs), ('ensemble-kalman_tracker', kp, coord)],
        color='k', linewidth=2, label='EKS',
    )
    if coord == 'x':
        ax.legend()

plt.suptitle(f'EKS results for {kp} ({view} view)', fontsize=14)
plt.tight_layout()

save_file = os.path.join(save_dir, 'example_eks_result.pdf')
plt.savefig(save_file)
plt.close()
print(f'see example EKS output at {save_file}')
