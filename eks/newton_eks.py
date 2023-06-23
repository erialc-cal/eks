import numpy as np
import os
import pandas as pd
import sys
from eks.utils import convert_lp_dlc

import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.notebook as tqdm
from scipy.optimize import *
np.random.seed(123)



### HELPER FUNCTIONS ###

def hessian(y, mu0, S0, A, B, ensemble_vars, E, plot = False):
    """ Calculates hessian for log p(Q|Y)
        obs : y_t|q_t = f(q_t)+ N(0, D_t) if linear, f(q) = B*q 
        latent : q_t|q_{t-1} = A*q_{t-1}+N(0,E)
        initial latent : q_0 = N(mu0, S0)
        
    Args:
        y: np.ndarray -- Each column is the vector of observations of keypoint
            shape (n_samples, n_keypoints)
        mu0: np.ndarray -- initial offset
            shape (n_latents)
        S0: np.ndarray -- initial covariance
            shape (n_latents, n_latents)
        A: np.ndarray -- Latent matrix
            shape (n_latents, n_latents)
        B: np.ndarray -- Observation matrix
            shape (n_keypoints, n_latents)
        ensemble_vars: np.ndarray -- ensemble variance 
            shape (n_samples, n_keypoints)
        E: np.ndarray -- latent variance
            shape (n_latents, n_latents)
        plot: boolean -- plots an excerpt of the hessian matrix as a heatmap
     """ 
    T = y.shape[0]
    r = mu0.shape[0]
    n = y.shape[1]
    D = np.zeros((n,n))
    invE = np.linalg.inv(E)
    H = np.zeros((T, T))

    for t in tqdm.tqdm(range(0,T-r)):
        D = np.diag(ensemble_vars[t])
        invD = np.linalg.inv(D)
        H[t:t+r, t:t+r] =  (B.T @ invD @ B + A.T @ invE @ A + invE)
        if t+2*r <= T:
            H[t:t+r, t+r:t+2*r] = - (A.T @ invE)
            H[t+r:t+2*r, t:t+r] = -(A.T @ invE).T  

    H[T-r:, T-r:] = invE + B.T @ invD @ B
    H[:r,:r] = np.linalg.inv(S0)+ A.T@invE@A
    print("Hessian")
    if plot :
        sns.heatmap(H[1980:,1980:])
    return H


def gradient(q, y, mu0, S0, A, B, ensemble_vars, invE):
    T = y.shape[0]
    G = np.zeros((T,r))
    D = np.zeros((y.shape[1], y.shape[1]))
    G[0,:] = A.T @ invE @(q[1,:] - A @ q[0,:])+np.linalg.inv(S0)@(q[0,:]-mu0)+B.T@(y[0,:]-B@q[0,:])
    
    for t in range(1,T-1):
        D = np.diag(ensemble_vars[t])
        invD = np.linalg.inv(D)
        G[t,:] = A.T @ invE@(q[t+1,:]-A@q[t,:]) - invE @(q[t,:]-A@q[t-1,:])+B.T @invD @(y[t,:]-B@q[t,:])
    G[T-1,:] =  - invE @(q[T-1,:]-A@q[T-2,:])+B.T @invD @(y[T-1,:]-B@q[T-1,:])
    
    return G

def schur_diag(H,k,l):
    """ Schur inversion lemma diagonal blocks 
    
    We need to invert H = ([A,B],[C,D]) with square blocks A of dimension k by k
    and D of dimension l by l and extract the inverted matrix's diagonal blocks 
    using Schur complement D'=(D-CA^{-1}B)^{-1}
    
    Trick: Computationally less expensive use Woodbury's lemma 
    (D-CA^{-1}B)^{-1} = D^{-1} + D^{-1}C(A -BD^{-1}C)^{-1}BD^{-1}
    
    H: np.ndarray -- Matrix of interest
        shape (k+l, k+l)
    k: int64 -- gives the top left block's dimensions
    l: int64 -- gives the bottom left block's dimensions
    """

    assert H.shape[0] == k+l
    M = np.zeros((k+l,k+l))
    A = H[:k,:k]
    D = H[k:k+l,k:k+l]
    B = H[:k,k:k+l]
    C = H[k:k+l, :k]
    invD = np.linalg.inv(D)
    #print(A-B@invD@C)
    if (np.linalg.matrix_rank(A - B@invD@C)== np.linalg.matrix_rank(A)):
        M[:k,:k] = np.linalg.inv(A-B@invD@C)
        M[k:k+l,k:k+l] = invD+invD@C@M[:k,:k]@B@invD
  
        return M
    else:
        print("Singular matrix")

        
        ####### M A I N #########
    



def kalman_newton_recursive(y, mu0, S0, A, B, ensemble_vars, E, max_iter=1):
    """
    One-pass Kalman recursive method described in J. Humphrey et J. West, "Kalman filtering with Newton's method" 
    https://math.byu.edu/~jeffh/publications/papers/HW.pdf 
    
    
    """
    r = mu0.shape[0]
    T = y.shape[0]
    
    invE = np.linalg.inv(E)
    grad_z = dict.fromkeys(range(0, T))
    q = np.zeros((T,r))
    q[0,:] = mu0
    qnew = q
    loss = np.zeros(max_iter)

    P = np.linalg.inv(S0) 
    for iter in range(max_iter):
        for t in range(1,T-1):
            D = np.diag(ensemble_vars[t])
            invD = np.linalg.inv(D)
            P = np.linalg.inv(np.linalg.inv(E+A@P@A.T)+B.T@invD@B)
            qnew[t,:] = A@q[t-1,:]-P@B.T@invD@(B@A@q[t-1,:]-y[t,:])   

        D = np.diag(ensemble_vars[T-1])
        invD = np.linalg.inv(D)
        P = np.linalg.inv(np.linalg.inv(E+A@P@A.T)+B.T@invD@B)
        qnew[T-1,:] = A@q[T-2,:]-P@B.T@invD@(B@A@q[T-2,:]-y[T-1,:])  
        loss[iter] = np.linalg.norm(qnew-q, 2)
        q = qnew
    if max_iter == 1:
        return q
    return q, loss
    

    
            ### PLOTS ###
        
def latent_plots(q_test, q, mean_array, n=200):
    fig, ax = plt.subplots(3,1,figsize=(20,12))
    for i in range(3):
        a = mean_array[i]
        ax[i].plot(q[:n,i], "-.",color="grey", label="reference eks")
        ax[i].plot(q_test[:,i]+a, "--",color="green", label="test eks")
    plt.legend()
    plt.suptitle("latent predictions on pupil data")
    
def opti_plots(y_test, y, n=200):
    fig, ax = plt.subplots(figsize=(20,4))
    ax.plot(y, "-.",color="grey", label="reference eks")
    ax.plot(y_test, "--",color="green", label="test eks")
    plt.legend()
    plt.suptitle("Truth predictions on pupil data")