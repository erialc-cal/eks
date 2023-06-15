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



### HELPER FUNCTIONS ###

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
        H[:r, :r] =  (B.T @ invD @ B + A.T @ invE @ A + invE)
        H[r:2*r, r:2*r] =  (B.T @ invD @ B + A.T @ invE @ A + invE)
    else:
        H[:r, :r] = (df2.T @ invD @ (f - y)+ df.T @ invD @ df)
        H[r:2*r, r:2*r] = (df2.T @ invD @ (f - y)+ df.T @ invD @ df)
    H[:r, r:2*r] = - A.T @ invE
    H[r:2*r, :r] = -(A.T @ invE).T  
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


def kalman_newton_greedy(y, mu0, A, B, D, E, f = None, df = None,df2 = None):
    """
    One-pass Kalman recursive method described in J. Humphrey et J. West, "Kalman filtering with Newton's method" 
    https://math.byu.edu/~jeffh/publications/papers/HW.pdf 
    
    
    """
    r = mu0.shape[0]
    T = y.shape[0]
    
    invE = np.linalg.inv(E)
    invD = np.linalg.inv(D)
    grad_z = dict.fromkeys(range(0, T))
    q = np.zeros((T,r))
    q[0,:] = mu0


    if f == None:
        print("Linear solve...")
        
        H = hessian(y, mu0, A, B, invD, invE, compressed=True)
        
        P = schur_diag(H,r,r)[r:2*r, r:2*r] # select bottom right block
        for t in tqdm.tqdm(range(1,T-1)):
           # z[t,:] = np.vstack([z[t-1,:], A@q[t-1,:]]) 
           # grad_z[t] = np.vstack([0, B.T@invD@(B@A@q[t,:]-y[t,:])])
           # Newton step : z = z - (H_new)^{-1} grad_z 
           # print(q[t,:].shape, P.shape, B.T.shape)
            q[t,:] = A@q[t-1,:]-P@B.T@invD@(B@A@q[t-1,:]-y[t,:])   

            F = np.hstack([np.zeros((3, 3*t)),A.T])
            # print((H + F.T @invE @ F).shape, (F.T @ invE).shape, (B.T@invD@B).shape)
            H = np.block([[H + F.T @invE @ F, - F.T @ invE],[-( F.T @ invE).T , invE + B.T @ invD @B]])

            P = schur_diag(H, len(H)-r,r)[len(H)-r:, len(H)-r:]       

        q[T-1,:] = A@q[T-2,:]-P@B.T@invD@(B@A@q[T-2,:]-y[T-1,:])  
    else: 
        # to be done, gradient is less straightforward
    return q
    



    
def kalman_newton_recursive(y, mu0, A, B, D, E, f = None, df = None,df2 = None):
    """
    One-pass Kalman recursive method described in J. Humphrey et J. West, "Kalman filtering with Newton's method" 
    https://math.byu.edu/~jeffh/publications/papers/HW.pdf 
    
    
    """
    r = mu0.shape[0]
    T = y.shape[0]
    
    invE = np.linalg.inv(E)
    invD = np.linalg.inv(D)
    grad_z = dict.fromkeys(range(0, T))
    q = np.zeros((T,r))
    q[0,:] = mu0


    if f == None:
        print("Linear solve...")
        
        # H = hessian(y, mu0, A, B, invD, invE, compressed=True),r,r
        P = schur_diag(hessian(y, mu0, A, B, invD, invE, compressed=True),r,r)[r:2*r, r:2*r] # select bottom right block
        for t in tqdm.tqdm(range(1,T-1)):
           # idea is to define z[t,:] = np.vstack([z[t-1,:], A@q[t-1,:]]) 
           # so grad_z[t] = np.vstack([0, B.T@invD@(B@A@q[t,:]-y[t,:])])
           # Newton step : z = z - (Hessian)^{-1} grad_z 
           # where hessian is block tridiagonal 
           # can be simplified using Schur, Woodbury and tridiagonal 
           # structure to the following updates
            q[t,:] = A@q[t-1,:]-P@B.T@invD@(B@A@q[t-1,:]-y[t,:])   

            P = np.linalg.inv(np.linalg.inv(E+A@P@A.T)+B.T@D@B)
            
        q[T-1,:] = A@q[T-2,:]-P@B.T@invD@(B@A@q[T-2,:]-y[T-1,:])  
    #else: 
        # to be done, gradient is less straightforward
    return q
    
    