import numpy as np
import os
import pandas as pd
import sys
from eks.utils import convert_lp_dlc
from eks.multiview_pca_smoother import ensemble_kalman_smoother_multi_cam

import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.notebook as tqdm
from scipy.optimize import *
np.random.seed(123)


def hessian(y, mu0, A, B, invD, invE, f = None, df = None, df2 = None, compressed=False):
    """ Calculates hessian for log p(Q|Y)
        obs : y_t|q_t = f(q_t)+ N(0, D) if linear, f(q) = B*q if not df = f'(q), df2 = f''(q)
        latent : q_t|q_{t-1} = A*q_{t-1}+N(0,E)
        
    Args:
        y: np.ndarray -- Each column is the vector of observations of keypoint
            shape (n_samples, n_keypoints)
        mu0: np.ndarray
            shape (n_latents)
        A: np.ndarray
            shape (n_latents, n_latents)
        B: np.ndarray
            shape (n_keypoints, n_latents)
        invD: np.ndarray
            shape (n_keypoints, n_keypoints)
        invE: np.ndarray
            shape (n_latents, n_latents)
        f: np.ndarray
            shape (n_samples, n_keypoints)
        df: np.ndarray
            shape (n_samples, n_keypoints)
        df2: np.ndarray
            shape (n_samples, n_keypoints)
     """ 
    T = y.shape[0]
    r = m0.shape[0]
    H = np.zeros(shape=(2*r,2*r))  # Hessian should be a block tridiagonal matrix with T blocks of size r by r but we can store the submatrix of size 2*r by 2*r since Kalman equations are quadratic
    
    if f == None: 
        # linear observation map
        H[:r, :r] = - (B.T @ invD @ B + A.T @ invE @ A + invE)
        H[r:2*r, r:2*r] = - (B.T @ invD @ B + A.T @ invE @ A + invE)
    else:
        H[:r, :r] = - (df2.T @ invD @ (f - y)+ df.T @ invD @ df)
        H[r:2*r, r:2*r] = - (df2.T @ invD @ (f - y)+ df.T @ invD @ df)
    H[:r, r:2*r] = A.T @ invE
    H[r:2*r, :r] = (A.T @ invE).T  
    if compressed == True:
        return H
    else :
        bigH = np.zeros((T, T))
        for i in range(T//r -1):
            bigH[r*i:r*(i+1), r*i:r*(i+1)] = H[:r,:r] # diagonal 
            bigH[r*(i+1):r*(i+2), r*i:r*(i+1)] = H[:r,r:2*r] # superdiagonal
            bigH[r*i:r*(i+1), r*(i+1):3*(i+2)] = H[r:2*r,:r] # subdiagonal
        bigH[T-2*r:T,T-2*r:T] = H
        print("Full matrix of size T by T")
        return bigH
    
    
    
    
def gradient(y, mu0, S0, q, A, B, invE, invD, f = None, df = None, df2 = None):
    """ Calculates hessian for log p(Q|Y)
    Linear is assumed as default, if non linear observation map, use f, df, df2
    
    Args:
        y: np.ndarray
            shape (n_samples, n_keypoints)
        mu0: np.ndarray
            shape (n_latents)
        q : np.ndarray -- latent variables obtained from mu0 and our Markov chain
            shape (n_latents, n_samples)
        A: np.ndarray
            shape (n_latents, n_latents)
        B: np.ndarray
            shape (n_keypoints, n_latents)
        invD: np.ndarray
            shape (n_keypoints, n_keypoints)
        invE: np.ndarray
            shape (n_latents, n_latents)
        f: np.ndarray
            shape (n_samples, n_keypoints)
        df: np.ndarray
            shape (n_samples, n_keypoints)
        df2: np.ndarray
            shape (n_samples, n_keypoints)
     """ 

    T = y.shape[0]
    r = mu0.shape[0]
    G = np.zeros(shape=(T, r))

    for i in range(1,T-1):
        if f == None:
            # linear map
            G[i,:] = - invE @(q[:,i] - A@q[:,i-1])+ A.T @ invE @(q[:,i+1]-A@q[:,i]) + B.T @ invD @ (y[i,:] - B @ q[:,i])
        else :
            G[i,:] = - invE @(q[:,i] - A@q[:,i-1])+ A.T @ invE @(q[:,i+1]-A@q[:,i]) + df.T @ invD @ (f[:,i]-q[:,i])
    # Boundary condition
    G[0, :] = A.T @ invE @ (q[:,1]-A@q[:,0])
    G[T-1,:] =  - invE @(q[:,T-1] - A@q[:,T-2])+ A.T @ invE @(-A@q[:,T-1]) + B.T @ invD @ (y[T-1,:] - B @ q[:,T-1])
    return G





def obj_loglikelihood(y,q, invE, invD, A, B, f= None):
    """ 
    Maximising loglikelihood in linear case
    
    hat{Q} = argmax_q log p(Q|O) = argmax_Q log p(Q) + log p(O|Q)
    with log p(Q) = -frac{1}{2} sum_{tk} (q_{tk} - q_{t+1,k})^T E_k^{-1} (q_{tk} - q_{t+1,k}) 
    and log p(O|Q) =-frac{1}{2} sum_{tkv} (f_{v}(q_{tk}) - O_{tkv})^T D_{tkv}^{-1} (f_v(q_{tk}) - O_{tkv})
       Args:
        y: np.ndarray
            shape (n_samples, n_keypoints)
        q: np.ndarray
            shape (n_latents)
        A: np.ndarray
            shape (n_latents, n_latents)
        B: np.ndarray
            shape (n_keypoints, n_latents)
        invD: np.ndarray
            shape (n_keypoints, n_keypoints)
        invE: np.ndarray
            shape (n_latents, n_latents)
    """
    T = y.shape[0]
    
    if f == None:
        diffq = np.sum(q[:,1:]-A@q[:,:T-1],axis=1)+ q[:,0]
        diffo = np.sum(y.T - B@q, axis=1)
        #print(diffo)
    else:
        diffq = np.sum(q[:,1:]-A@q[:,:T-1],axis=1)+ q[:,0]
        diffo = np.sum(y.T - f, axis=1)
    loglikelihood = - 0.5*(diffq.T @ invE @ diffq) - 0.5*(diffo.T @ invD @ diffo) 
    return loglikelihood


####### M A I N #########

def newton_linear(y,mu0, S0, q, A, B, invE, invD, max_iter=1000, eps = 0.01):
    H_1 = np.linalg.inv(hessian(y, mu0, A, B, invD, invE))
    old = q.T
    loss = [obj_loglikelihood(y, old.T, invE, invD, A, B)]
    
    for it in tqdm.tqdm(range(max_iter)):
        G = gradient(y, mu0, S0, old.T, A, B, invE, invD)

        new = old - H_1@ G
        obj = obj_loglikelihood(y,new.T,invE,invD, A, B)
        loss.append(obj)
        if np.linalg.norm(new - old) < eps:
            print("Local maximum reached before maximum iterations reached")
            break
            return old 
        else:
            old = new
            
    return new,loss
