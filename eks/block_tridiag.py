import numpy as np


def Block_thomas_algo(y, m0, S0, A, B, E, ensemble_vars):
    D = ensemble_vars[:,0]
    M = np.zeros((r,r))
    M = np.linalg.inv(S0) + A.T @ invE @ A 
    invM = np.linalg.inv(M)
    Tr = np.zeros((T,r,r)) 
    q = np.zeros((T,r))
    s = np.zeros((T,r))
    Tr[0,:] = invM@ A.T @ invE
    G = gradient_tridiag(y, m0, S0, A, B, E, ensemble_vars)
    q[0,:] = - invM @ G[:,0]
    for t in range(1,T):
        D = np.diag(ensemble_vars[:,t])
        invD = np.linalg.inv(D)
        M = invE + A.T @ invE @ A + B.T @ invD @ B - A.T@invE@invM@(A.T @ invE).T
        invM = np.linalg.inv(M)   
        Tr[t,:] = invM @ A.T @ invE
        q[t,:] = - invM @( G[:,t] - (A.T@invE).T @q[t-1,:] )
    s[T-1,:] = q[T-1,:]
    for t in range(T-2, -1, -1):
        s[t,:] = q[t,:] + Tr[t,:]@s[t+1,:]
        
    return s


def gradient_tridiag(y, m0, S0, A, B, E, ensemble_vars):
    r = m0.shape[0]
    T = y.shape[0]
    G = np.zeros((r,T))
    q = np.zeros((T,r))
    invE = np.linalg.inv(E)
    D = np.diag(ensemble_vars[:,0])
    invD = np.linalg.inv(D)

    for i in range(T-1):
        D = np.diag(ensemble_vars[:,i])
        invD = np.linalg.inv(D)
        G[:,i] = - B.T @ invD@(y[i,:] - B@q[i,:]) - A.T @ invE @(q[i+1,:]-A@q[i,:]) + invE@(q[i,:]-A@q[i-1,:])
    G[:,T-1] = - B.T @ invD@(y[T-1,:] - B@q[T-1,:])+ invE@(q[T-1,:]-A@q[T-2,:])
    return G



    
    
    