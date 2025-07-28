import numpy as np
import ot

def compute_unbalanced_ot(df1, df2, reg_m, reg_kl):
    """
    Computes unbalanced optimal transport plan between two datasets.
    
    Parameters:
    df1: array-like, shape (n, d)
    df2: array-like, shape (m, d)
    reg_m: float, regularization term for mass preservation
    reg_kl: float, regularization term for KL divergence
    
    Returns:
    transport_plan: numpy.ndarray, transport plan matrix (n x m)
    """
    # Convert dataframes to numpy arrays
    X = np.array(df1)
    Y = np.array(df2)
    
    # Uniform weights for now, adjust as necessary
    a = np.ones(X.shape[0]) / X.shape[0]
    b = np.ones(Y.shape[0]) / Y.shape[0]
    
    # Cost matrix (Euclidean distances)
    M = ot.dist(X, Y, metric='euclidean')
    
    # Unbalanced OT
    transport_plan = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, reg_m, reg_kl)
    
    return transport_plan
