# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import pandas as pd
import sys
from eks.utils import *
from eks.multiview_pca_smoother import pca, ensemble_kalman_smoother_multi_cam
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
from scipy.optimize import *
from scipy.interpolate import interp1d
from eks.ensemble_kalman import ensemble, filtering_pass, kalman_dot, smooth_backward


#%%%%


camera_names = ['main', 'top', 'right']
keypoint_ensemble_list = ['mid','fork','chin_base']
tracker_name = 'heatmap_mhcrnn_tracker'
num_cameras = len(camera_names)
labeled_data = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/CollectedData_new.csv", header = [1,2], index_col=0)

#%% set distance constraint projection for a few q's in keypoint_ensemble_list


camkeys = ["_main","_top","_right"]
# flatten columns
labeled_data.columns = ['_'.join(tup).rstrip('_') for tup in labeled_data.columns.values]
mask = []
for keys in keypoint_ensemble_list:
    for cam in camkeys:
        for coord in ["_x", "_y"]:
            mask.append(keys+cam+coord)
            


#%%% 

# LABELED DATA 

markers_list = labeled_data.reset_index()[mask]
# Ensemble
scaled_dict = []
good_frames_dict = []
good_preds_dict = []
ensemble_vars_dict = []
means_camera_dict = []
markers_list_cameras  = []
            
n = len(markers_list)
nkeys = len(keypoint_ensemble_list)
num_cameras = len(camkeys)
y_obs = np.empty((nkeys, n, 2*num_cameras))
q = np.empty((nkeys, n, 3))
for j, keypoint_ensemble in enumerate(keypoint_ensemble_list):
    markers_list_cameras = [[] for i in range(num_cameras)]
    for i,cam in enumerate(camkeys):
        tmp = []

        for m in markers_list.keys():
            if cam in m and keypoint_ensemble in m:
                tmp.append(markers_list[m])
    
        markers_list_cameras[i].append(pd.concat(tmp, axis=1))

    
    y = np.asarray(markers_list_cameras).reshape(n, nkeys*2)
    
    # fill nans by median value 
    col_mean = np.nanmedian(y, axis=0)
    inds = np.where(np.isnan(y))
    y[inds] = np.take(col_mean, inds[1])
    
    # scale 
    from sklearn.preprocessing import scale
    scaled_y = scale(y)
    # get PCA 
    labeled_pca, labeled_var = pca(y, 3)
    
    q[j,:,:] = labeled_pca.transform(scaled_y)
    y_obs[j,:,:] = y

#%%%


# Initialize L as a lower triangular matrix
n = 3  # Define the size of L
L_initial = np.tril(np.eye(n)).flatten()
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

L = find_linear_transformation(q, L_initial)

pair_list = pairwise(keypoint_ensemble_list)
v = variance_limb_plot(5, L, q, pair_list)

pair_list = [('mid', 'chin_base'),('chin_base','fork')]
D = get_3d_distance_loss(q, L_initial, keypoint_ensemble_list,pair_list, 3)





#%%% Ensembling keypoint per keypoint

def ensembling_multiview(markers_list_cameras, keypoint_ensemble, smooth_param, quantile_keep_pca, camera_names, plot=True):
# -------------------------------------------------------------
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

        
    if plot:
        test = cam_ensemble_preds
        # print(len(test[0]))
        plt.imshow(im)
        plt.scatter(test[1][:,0], test[1][:,1], color='green')
        plt.scatter(test[0][:,0], test[0][:,1], color='blue', alpha=0.7)
        plt.scatter(test[2][:,0], test[2][:,1], color='red', alpha=0.7)
        plt.show()
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
    scaled_ensemble_preds = remove_camera_means(ensemble_preds[None,:,:], means_camera)[0]
    
    return scaled_ensemble_preds, good_frames, good_scaled_ensemble_preds, ensemble_vars,means_camera


#%%% Stacking up ensembled data for PCA

def multiview_pca_bodyparts(scaled_dict,good_preds_dict,good_frames_dict):
    n, T, v= np.shape(scaled_dict)
    stacked_preds = np.vstack(scaled_dict)
    stacked_good_preds = np.vstack(good_preds_dict)
    good_ensemble_pcs= {key: None for key in range(len(good_frames_dict))}
    # PCA
    
    ensemble_pca, ensemble_ex_var = pca(stacked_good_preds, 3)
    ensemble_pcs = ensemble_pca.transform(stacked_preds)
    # unstack
    ensemble_pcs = ensemble_pcs.reshape(n, T, 3)
    for key in range(len(good_frames_dict)):
        
    
        good_ensemble_pcs[key] = ensemble_pcs[key][good_frames_dict[key]]

    
    return scaled_dict,ensemble_pca,ensemble_ex_var,ensemble_pcs,good_ensemble_pcs
    



#%%

from scipy.optimize import minimize

# Minimise \sum_ij Var(||L(q_ti-q_tj)||_2)/\sum_k Var(Lq_tk)

def objective(L):
    s = 0
    d = 0
    n = int(np.sqrt(len(L)))
    L = np.asarray(L).reshape((n,n))
    upper_indices = np.triu_indices(n,1) #offset to diagonal
    # (n-1)n/2 
    L[upper_indices] = np.zeros((n*(n-1)//2)) # constraint upper triangle to zeros
    for i in range(q.shape[0]):
        d += np.var(L@q[i,:,:].T)
        for j in range(q.shape[0]):
            if j != i:
                s+= np.var(np.linalg.norm(L@(q[i,:,:]-q[j,:,:]).T,axis=0))
    return s/d

# L = np.tri(3,3,0)
# objective(L,q)



def find_linear_transformation(q, L_initial):
    
    # Define the optimization problem with the objective function and constraint
    problem = {
        'fun': objective,
        'x0': L_initial
        }
    
    # Solve the optimization problem
    result = minimize(**problem)
    
    # Get the optimal solution for L
    optimal_L = result.x
    
    return optimal_L.reshape((n,n))
    



#%%

def pairwise(t):
    return [(a, b) for idx, a in enumerate(t) for b in t[idx + 1:]]

def variance_limb(L,q,frames_start=0, frames_end=10):
    d = 0
    s = np.zeros((q.shape[0],q.shape[0]))
    for i in range(q.shape[0]):
        d += np.var(L@q[i,:,:].T)
        for j in range(q.shape[0]):
            if j != i:
                s[i,j]= np.var(np.linalg.norm(L@(q[i,frames_start:frames_end,:]-q[j,frames_start:frames_end,:]).T,axis=0))
    return s/d
    

def variance_limb_plot(num_frames, L, q, pair_list):
    # pair_list = pairwise(keys)
    tot_var =  variance_limb(L,q,frames_start =0,frames_end = q.shape[1])
    var_dict = {}
    for key_pair in pair_list:
        i= keypoint_ensemble_list.index(key_pair[0])
        j=keypoint_ensemble_list.index(key_pair[1])
        var_list = []
        for frame in range(num_frames,q.shape[1]):
            s = variance_limb(L,q,frames_start = frame-num_frames,frames_end=frame)
            var_list.append(s[i,j]/tot_var[i,j]*100)
        var_dict[key_pair] = var_list[1:]
        plt.plot(var_dict[key_pair], label = '{}'.format(key_pair))
    plt.legend()
    plt.title('Variance proportion of limb distance over {}'.format(num_frames)+' frames')
    return var_dict    

# fig, ax =plt.subplots(1,2)
# sns.heatmap(variance_limb(L.reshape(3,3),q),ax=ax[0], vmin=0, vmax=15,cmap="PiYG")
# sns.heatmap(variance_limb(L_initial.reshape(3,3),q),ax=ax[1], vmin=0, vmax=15,cmap="PiYG")
# ax[0].set_title("minimiser")
# ax[1].set_title("initial guess")


# df = pd.DataFrame({'x':(L@np.vstack(q).T)[0,:],'y':(L@np.vstack(q).T)[1,:],'z':(L@np.vstack(q).T)[2,:],'limb':np.repeat(keypoint_ensemble_list,51)
#                    })

# fig= px.line_3d(df, x='x', y='y',z='z',color='limb')
# fig.show()



# df = pd.DataFrame({'x':(np.vstack(q).T)[0,:],'y':(np.vstack(q).T)[1,:],'z':(np.vstack(q).T)[2,:],'limb':np.repeat(keypoint_ensemble_list,51)
#                    })

# fig= px.line_3d(df, x='x', y='y',z='z',color='limb')
# fig.show()




#%% 

# Graph of the forme [('mid','chin'),('fork','chin')]

# if full graph can use pairwise(keypoint_ensemble_list)


def get_3d_distance_loss(q, L_initial, keypoint_ensemble_list, constrained_keypoints_graph, num_cameras):

    n = len(keypoint_ensemble_list)
    T = np.vstack(q).shape[0]//n
    
    D = np.zeros((T, n,n))
    L = find_linear_transformation(q, L_initial)
    new_q = (L@np.vstack(q).T).reshape((n,T,num_cameras))
    
    # get constrained distances
    for t in range(T):
        for keypair in constrained_keypoints_graph:
            i = keypoint_ensemble_list.index(keypair[0])
            j = keypoint_ensemble_list.index(keypair[1])
            D[t,i,j]= np.linalg.norm(new_q[i,t,:]-new_q[j,t,:])
            D[t,j,i]= np.linalg.norm(new_q[i,t,:]-new_q[j,t,:])
            
    return D

# ASSUME WE STACK BODYPARTS ONE AFTER THE OTHER SO IF 3 BODYPARTS AND t = 51, q IS OF SHAPE (153,3)

def filtering_pass_with_constraint(y, m0, S0, C, R, A, Q, ensemble_vars, D,L, keypoint_ensemble_list, constrained_keypoints_graph=None, mu=1):
    if constrained_keypoints_graph == None:
        constrained_keypoints_graph = pairwise(keypoint_ensemble_list)
        # all nodes are connected from bodyparts of interest
    # y.shape = (keypoints, time steps, views) 

    T = y.shape[1]  # number of time stpes
    n = len(keypoint_ensemble_list) # number of keypoints
    v = y.shape[2] # number of views
    mf = np.zeros(shape=(n,T, m0.shape[0]))
    Vf = np.zeros(shape=(n,T, m0.shape[0], m0.shape[0]))
    S = np.zeros(shape=(n,T, m0.shape[0], m0.shape[0]))
    for k in range(n):
        for i in range(v):
           R[i,i] = ensemble_vars[k][0][i]
        mf[k,0] =m0 + kalman_dot(y[k,0, :] - np.dot(C, m0), S0[k], C, R)
        Vf[k,0, :] = S0[k] - kalman_dot(np.dot(C, S0[k]), S0[k], C, R)
        S[k,0] = S0[k]
    for k, part in enumerate(keypoint_ensemble_list):
        for i in range(1,T):
           
            for t in range(v):
                R[t,t] = ensemble_vars[k][i][t]
            S[k,i-1] = np.dot(A, np.dot(Vf[k,i-1, :], A.T)) + Q
            #print(S[i-1], )
            y_minus_CAmf = y[k,i, :] - np.dot(C, np.dot(A, mf[k,i-1, :])) 
            grad = gradient_distance(mf@L, part, keypoint_ensemble_list, constrained_keypoints_graph)
            #print(np.linalg.norm(grad))
            hess = hessian_distance(mf@L, part,  keypoint_ensemble_list, constrained_keypoints_graph)
            if any(part in i for i in constrained_keypoints_graph):
                # add gradient and hessian penalisaiton
                mf[k,i, :] = np.dot(A, mf[k,i-1, :]) + kalman_dot(y_minus_CAmf, S[k,i-1], C, R) + mu*grad[i//len(keypoint_ensemble_list),: ]
                # print(S[k,i-1], np.linalg.inv(np.linalg.inv(S[k,i-1])))
                S[k,i-1] = np.linalg.inv(np.linalg.inv(S[k,i-1])+mu*hess)
                
            else:
                mf[k,i, :] = np.dot(A, mf[k,i-1, :]) + kalman_dot(y_minus_CAmf, S[k,i-1], C, R) 
            Vf[k,i, :] = S[k,i-1] - kalman_dot(np.dot(C, S[k,i-1]), S[k,i-1], C, R)
            
    return mf, Vf, S     

    

def gradient_distance(q, part, keypoint_ensemble_list, constrained_keypoints_graph):
    # sum_nodes connected to part (q[part,:]-q[connected_part,:])/np.linalg.norm(----)
    p = keypoint_ensemble_list.index(part)
    n,T,v = q.shape
    neighbors = [item[0] for item in constrained_keypoints_graph if item[1] == part]+[item[1] for item in constrained_keypoints_graph if item[0] == part]
    nei_idx = []
    grad = np.zeros((T,v))
    for elem in neighbors:
        nei_idx.append(keypoint_ensemble_list.index(elem))  # get neighbor index
        
    for idx in nei_idx:
        grad += (q[p,:, :] - q[idx ,:, :])/np.linalg.norm(q[p,:, :] - q[idx,:, :])
        
    return grad
    
    
def hessian_distance(q, part, keypoint_ensemble_list, constrained_keypoints_graph):
    p = keypoint_ensemble_list.index(part)
    neighbors = [item[0] for item in constrained_keypoints_graph if item[1] == part]+[item[1] for item in constrained_keypoints_graph if item[0] == part]
    nei_idx = []
    n = len(keypoint_ensemble_list)
    hess = np.zeros((n,n))
    for elem in neighbors:
        nei_idx.append(keypoint_ensemble_list.index(elem))  # get neighbor index
    
    for idx in nei_idx:
        if (np.linalg.norm(q[p,:, :] - q[idx,:, :])>0):
            hess += 1/(np.linalg.norm(q[p,:, :] - q[idx,:, :])) - 1/(np.linalg.norm(q[p,:, :] - q[idx,:, :])**3) \
                * (q[p,:, :] - q[idx,:, :]).T @(q[p,:, :] - q[idx,:, :])
        else:
            print("Hessian not well defined")
    return hess



#%%%%% TEST 
# q = q.reshape(q.shape[0]*q.shape[1],q.shape[2])
# y_obs = y_obs.reshape(y_obs.shape[0]*y_obs.shape[1],y_obs.shape[2])


# ##### Set values for kalman filter #####
# m0 = np.asarray([0.0, 0.0, 0.0]) # initial state: mean
# S0 =  np.asarray([[np.var(q[:,0]), 0.0, 0.0], [0.0, np.var(q[:,1]), 0.0], [0.0, 0.0, np.var(q[:,2])]]) # diagonal: var

# A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) #state-transition matrix,
# # Q = np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]) #state covariance matrix?????
# d_t = q[1:] - q[:-1]

# Q = smooth_param*np.cov(d_t.T)

# C = labeled_pca.components_.T # Measurement function is inverse transform of PCA
# R = np.eye(labeled_pca.components_.shape[1]) # placeholder diagonal matrix for ensemble variance




# print(f"filtering ...")
# mf, Vf, S= filtering_pass_with_constraint(y_obs, m0, S0, C, R, A, Q, ensemble_vars, D, keypoint_ensemble_list, constrained_keypoints_graph=None, mu=1)


# y_m_filt = np.dot(C, mf.T).T
# y_v_filt = np.swapaxes(np.dot(C, np.dot(Vf, C.T)), 0, 1)

# # Do the smoothing step
# print(f"smoothing {keypoint_ensemble}...")
# ms, Vs, _ = smooth_backward(y_obs, mf, Vf, S, A, Q, C)
# print("done smoothing")

# # Smoothed posterior over y
# y_m_smooth = np.dot(C, ms.T).T
# y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)















#%% Test

folder = "/eks_opti"
operator = "/20210204_Quin/"
name = "img197707"

baseline = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions/eks"+operator+name+".csv", header=[ 1, 2],index_col=0)
#new = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/one-video-mirror-fish-predictions"+folder+operator+name, header=[ 1, 2], index_col=0)
baseline0 = pd.read_csv("/Users/clairehe/Documents/GitHub/eks/data/misc/mirror-fish_ensemble-predictions/eks"+operator+name+".csv", header=[0, 1, 2],index_col=0)


# NOTE! replace this path with an absolute path where you want to save EKS outputs
eks_save_dir = '/Users/clairehe/Documents/GitHub/eks/data/misc/one-video-mirror-fish-predictions/eks_opti/'

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


#    'head', 'chin_base', 'chin1_4', 'chin_half','chin3_4', 'chin_tip', 'mid', 'fork',
#   'stripeA', 'stripeP', 'tail_neck', 'dorsal', 'anal', 'caudal_d', 'caudal_v',

image_path = "/Users/clairehe/Documents/GitHub/eks/data/mirror-fish/labeled-data"
im = plt.imread(image_path+operator+name+".png")
plt.imshow(im)
plt.suptitle("labeled "+name)


session = '20210204_Quin'
frame = 'img197707.csv'
smooth_param = 0.01
quantile_keep_pca = 50
# Get markers list from networks
markers_list = []
for model_dir in model_dirs:
    csv_file = os.path.join(model_dir, session, frame)
    df_tmp = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
    keypoint_names = [l[1] for l in df_tmp.columns[::3]]
    markers_tmp = convert_lp_dlc(df_tmp, keypoint_names, model_name=tracker_name)
    markers_list.append(markers_tmp)

# Ensemble
scaled_dict = []
good_frames_dict = []
good_preds_dict = []
ensemble_vars_dict = []
means_camera_dict = []
for n, keypoint_ensemble in enumerate(keypoint_ensemble_list):
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
    # ENSEMBLING PER KEYPOINTS
    scaled_ensemble_preds, good_frames, good_scaled_ensemble_preds,ensemble_vars,means_camera = ensembling_multiview(markers_list_cameras, keypoint_ensemble, smooth_param, quantile_keep_pca, camera_names, plot=True)
    scaled_dict.append(scaled_ensemble_preds)
    good_frames_dict.append(good_frames)
    good_preds_dict.append(good_scaled_ensemble_preds)
    ensemble_vars_dict.append(ensemble_vars)
    means_camera_dict.append(means_camera)
ensemble_vars = np.array(ensemble_vars_dict)


stacked_preds,ensemble_pca,ensemble_ex_var,ensemble_pcs,good_ensemble_pcs =  multiview_pca_bodyparts(scaled_dict,good_preds_dict,good_frames_dict)

y_obs = np.asarray(stacked_preds)

#compute center of mass
#latent variables (observed)
good_z_t_obs = good_ensemble_pcs #latent variables - true 3D pca

n,T, v = y_obs.shape
nkeys = len(keypoint_ensemble_list)
##### Set values for kalman filter #####
m0 = np.asarray([0.0, 0.0, 0.0]) # initial state: mean
S0 = np.zeros((nkeys,m0.shape[0], m0.shape[0] ))
d_t = {key: None for key in range(nkeys)}
# need different variance for each bodyparts 
for k in range(nkeys):
    S0[k,:,:] =  np.asarray([[np.var(good_z_t_obs[k][:,0]), 0.0, 0.0], [0.0, np.var(good_z_t_obs[k][:,1]), 0.0], [0.0, 0.0, np.var(good_z_t_obs[k][:,2])]]) # diagonal: var
    d_t[k] = good_z_t_obs[k][1:] - good_z_t_obs[k][:-1]

    Q = smooth_param*np.cov(d_t[k].T)

A = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) #state-transition matrix,
# Q = np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]) #state covariance matrix?????


C = ensemble_pca.components_.T # Measurement function is inverse transform of PCA
R = np.eye(ensemble_pca.components_.shape[1]) # placeholder diagonal matrix for ensemble variance

#print(f"filtering {keypoint_ensemble}...")
#mf, Vf, S = filtering_pass(y_obs, m0, S0, C, R, A, Q,ensemble_vars)
print("constraint")

mfc, Vfc, Sc = filtering_pass_with_constraint(y_obs, m0, S0, C, R, A, Q,ensemble_vars, D, L,keypoint_ensemble_list, constrained_keypoints_graph=[('fork','mid'),('chin_base','fork')], mu=0)


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








