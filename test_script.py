import os
import pandas as pd
import numpy as np
import sys

#%%%%%%%%%% Set paths, requirements

#os.chdir('/Users/clairehe/Documents/GitHub/eks/')
#print(os.getcwd())
#!pip install -r requirements.txt
#!pip install -e .

#%%%%%%%%%%

from eks.utils import convert_lp_dlc
from eks.multiview_pca_smoother import ensemble_kalman_smoother_multi_cam


camera_names = ['main', 'top', 'right']
keypoint_ensemble_list = [
    'head', 'chin_base', 'chin1_4', 'chin_half','chin3_4', 'chin_tip', 'mid', 'fork',
    'stripeA', 'stripeP', 'tail_neck', 'dorsal', 'anal', 'caudal_d', 'caudal_v',
]
tracker_name = 'heatmap_mhcrnn_tracker'
num_cameras = len(camera_names)

# NOTE! replace this path with an absolute path where you want to save EKS outputs
eks_save_dir = '/Users/clairehe/Documents/GitHub/eks/data/misc/one-video-mirror-fish-predictions/eks_outputs/'

# path for prediction csvs
file_path = '/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions'

# NOTE! replace these paths with the absolute paths to prediction csvs on your local computer
model_dirs = [
    file_path+"/network_0",
    file_path+"/network_1",
    file_path+"/network_2",
    file_path+"/network_3",
    file_path+"/network_4",
]

sessions = os.listdir(model_dirs[0])
for session in sessions:
    frames = os.listdir(os.path.join(model_dirs[0], session))
    for frame in frames:
        #print(frame)
        # extract all markers
        markers_list = []
        for model_dir in model_dirs:
            csv_file = os.path.join(model_dir, session, frame)
            df_tmp = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
            keypoint_names = [l[1] for l in df_tmp.columns[::3]]
            markers_tmp = convert_lp_dlc(df_tmp, keypoint_names, model_name=tracker_name)
            markers_list.append(markers_tmp)
        # make empty dataframe to write results into
        df_eks = df_tmp.copy()
        for col in df_eks.columns:
            if col[-1] == 'likelihood':
                df_eks[col].values[:] = 1.0
            else:
                df_eks[col].values[:] = np.nan
        # fit kalman on a keypoint-by-keypoint basis
        for keypoint_ensemble in keypoint_ensemble_list:
            markers_list_cameras = [[] for i in range(num_cameras)]
            for m in markers_list:
                for camera in range(num_cameras):
                    markers_list_cameras[camera].append(
                        m[[key for key in m.keys() 
                           if camera_names[camera] in key 
                           and 'likelihood' not in key 
                           and keypoint_ensemble in key]
                         ]
                    )
            cameras_df = ensemble_kalman_smoother_multi_cam(
                markers_list_cameras=markers_list_cameras, 
                keypoint_ensemble=keypoint_ensemble, 
                smooth_param=0.01,
                quantile_keep_pca=50, 
                camera_names=camera_names,
            )
            # put results into new dataframe
            for camera in camera_names:
                df_tmp = cameras_df[f'{camera}_df']
                for coord in ['x', 'y']:
                    src_cols = ('ensemble-kalman_tracker', f'{keypoint_ensemble}', coord)
                    dst_cols = (tracker_name, f'{keypoint_ensemble}_{camera}', coord)
                    df_eks.loc[:, dst_cols] = df_tmp.loc[:, src_cols]
        # save eks results
        save_dir = os.path.join(eks_save_dir, session)
        os.makedirs(save_dir, exist_ok=True)
        df_eks.to_csv(os.path.join(save_dir, frame))