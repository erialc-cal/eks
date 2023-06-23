### ARCHIVE


def block_tri_solve(A, b, N):
    rows = A.shape[0]
    cols = A.shape[1]
    
    assert rows%N == 0
    
    n_bl= rows//N
    
    b = b.reshape((N, n_bl))
    
    x = np.zeros((N, n_bl))
    c = x
    
    # Diag
    D = np.zeros((N,N, n_bl))
    Q = D
    G = D
    
    #subdiag
    C = np.zeros((N,N, n_bl-1))
    
    # supdiag
    B = C
    
    for k in range(1,n_bl-1):
        # block = slice((k - 1) * N, k * N)
        # print(A[(k-1)*N:k*N, (k-1)*N:k*N].shape, D[:,:,k].shape)
        D[:,:,k] = A[(k-1)*N:k*N, (k-1)*N:k*N]
        # print(A[k*N:(k+1)*N, (k-1)*N:k*N].shape, B[:,:,k].shape)
        B[:,:,k] = A[k*N:(k+1)*N, (k-1)*N:k*N]
        C[:,:,k] = A[(k-1)*N:k*N, k*N:(k+1)*N]
    
    D[:,:,n_bl-1] = A[(n_bl - 1) * N: n_bl * N, (n_bl - 1) * N: n_bl * N]
    #print(Q.shape)
    Q[:,:,0] = D[:,:,0]
    G[:,:,0] = np.linalg.lstsq(Q[:,:,0],C[:,:,0])[0]
    
    for i in range(1, n_bl-1):
        Q[:,:,k]=D[:,:,k]-B[:,:,k-1]@G[:,:,k-1]
        G[:,:,k] = np.linalg.lstsq(Q[:,:,k], C[:,:,k])[0]
        
    Q[:,:,n_bl-1] = D[:,:,n_bl-1]-B[:,:,n_bl-2]@G[:,:,n_bl-2]
    c[:,0] = np.linalg.lstsq(Q[:,:,0], b[:,1])[0]
    
    for k in range(1, n_bl):
        c[:,k] = np.linalg.lstsq(Q[:,:,k],b[:,k]-B[:,:,k-1]@c[:,k-1])[0]
    x[:,n_bl-1] = c[:,n_bl-1]
    
    for k in range(1, n_bl-1,-1):
        x[:,k] = c[:,k]-G[:,:,k]@x[:,k+1]
    return x
    
        