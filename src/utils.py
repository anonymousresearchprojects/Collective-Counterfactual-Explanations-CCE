import random
import os
import numpy as np
import torch
import pandas as pd
# --------------------------
# Functions
# --------------------------

import numpy as np

def chi_squared_divergence(sample1, sample2, bins=4):
    """
    Calculate the chi-squared divergence between two 2D samples.

    Parameters:
        sample1: array-like, shape (n_samples1, 2)
            First sample of 2D observations.
        sample2: array-like, shape (n_samples2, 2)
            Second sample of 2D observations.
        bins: int or tuple of int
            Number of bins for the 2D histograms (same for both dimensions).
    
    Returns:
        chi_squared: float
            The chi-squared divergence.
    """
    # Create 2D histograms for both samples
    hist1, x_edges, y_edges = np.histogram2d(sample1.iloc[:, 0], sample1.iloc[:, 1], bins=bins, density=True)
    hist2, _, _ = np.histogram2d(sample2.iloc[:, 0], sample2.iloc[:, 1], bins=(x_edges, y_edges), density=True)

    # Flatten the histograms
    hist1 = hist1.flatten()
    hist2 = hist2.flatten()

    # Add a small constant to avoid division by zero or log of zero
    epsilon = 1e-10
    hist1 += epsilon
    hist2 += epsilon

    # Normalize histograms to sum to 1 (to form probability distributions)
    hist1 /= np.sum(hist1)
    hist2 /= np.sum(hist2)

    # Compute chi-squared divergence
    chi_squared = np.sqrt(np.sum((hist1 - hist2)**2 / hist2))
    return chi_squared


def pairwise_distances(df1, df2):
    # Extract coordinates as numpy arrays
    X1 = df1[['feature1', 'feature2']].to_numpy()  # shape (m, 2)
    X2 = df2[['feature1', 'feature2']].to_numpy()  # shape (n, 2)
    
    # Separate into x and y components for convenience
    X1_x = X1[:, 0].reshape(-1, 1)  # shape (m, 1)
    X1_y = X1[:, 1].reshape(-1, 1)  # shape (m, 1)
    X2_x = X2[:, 0].reshape(1, -1)  # shape (1, n)
    X2_y = X2[:, 1].reshape(1, -1)  # shape (1, n)

    # Compute the pairwise differences
    dx = X1_x - X2_x  # shape (m, n)
    dy = X1_y - X2_y  # shape (m, n)

    # Compute Euclidean distances
    D = np.sqrt(dx**2 + dy**2)  # shape (m, n)
    return D

def euclidean(data1, data2):
    """
    Calculate row-wise Euclidean distances between two Pandas DataFrames.
    
    Parameters:
        data1: pd.DataFrame
            First DataFrame.
        data2: pd.DataFrame
            Second DataFrame. Must have the same number of rows as data1.
    
    Returns:
        np.ndarray
            Array of Euclidean distances for each row.
    """
    # Check if both dataframes have the same number of rows
    if data1.shape[0] != data2.shape[0]:
        raise ValueError("The data frames must have the same number of rows.")
    
    # Convert dataframes to NumPy arrays for efficient computation
    array1 = data1.to_numpy()
    array2 = data2.to_numpy()
    
    # Calculate row-wise Euclidean distances
    euclidean_distances = np.sqrt(np.sum((array1 - array2) ** 2, axis=1))
    
    return euclidean_distances

def seed_everything(seed: int):
    """
    Set the seed for generating random numbers to ensure reproducibility.
    This function sets the seed for the following libraries:
    - random
    - os (for PYTHONHASHSEED)
    - numpy
    - torch (including CUDA)
    Args:
        seed (int): The seed value to set for random number generation.
    Returns:
        None
    """


    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_backend(method):
    if method in ['collective', 'wacher', 'roar', 'cchvae', 'growing_spheres', 'face', 'crud', 'dice', 'clue', 'actionable_recourse', 'cem', 'revise']:
        backend = 'pytorch'
    else:
        backend = 'sklearn'
    return backend

def print_in_box(message):
    """
    Prints the given message inside a box of # symbols.
    
    Args:
        message (str): The message to print in the box.
    """
    # Split the message into lines for multi-line support
    lines = message.split('\n')
    max_length = max(len(line) for line in lines)  # Find the length of the longest line

    # Print the top border of the box
    print('#' * (max_length + 4))

    # Print each line with padding
    for line in lines:
        print(f"# {line.ljust(max_length)} #")

    # Print the bottom border of the box
    print('#' * (max_length + 4))


def barycentric_projection(X, Y, pi):
    """
    Compute the barycentric projection map T from X to Y based on the transport plan pi.

    Parameters:
    - X: numpy array of shape (m, d), source points
    - Y: numpy array of shape (n, d), target points
    - pi: numpy array of shape (m, n), transport plan

    Returns:
    - T: numpy array of shape (m, d), mapped points in Y
    """
    # Ensure pi is a numpy array
    pi = np.array(pi)
    
    # Check dimensions
    m, n = pi.shape
    assert X.shape[0] == m, "Number of source points must match pi's number of rows."
    assert Y.shape[0] == n, "Number of target points must match pi's number of columns."
    
    # Normalize pi so that each row sums to 1
    row_sums = pi.sum(axis=1, keepdims=True)
    # To avoid division by zero, set zero sums to one (will result in zero mapping)
    row_sums[row_sums == 0] = 1
    pi_normalized = pi / row_sums
    
    # Compute the mapped points T(X) = pi_normalized @ Y
    T = pi_normalized @ Y  # Shape: (m, d)
    
    return T

def sqrt_column(df):
    df["feature1"] = np.log(df["feature1"].clip(lower=0))
    df["feature2"] = np.log(df["feature2"].clip(lower=0))
    return df


def filter_by_quantiles(df, columns, lower_quantile=0.1, upper_quantile=0.9):
    """
    Filters the DataFrame rows where the values in specified columns are 
    between the lower and upper quantiles.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to apply the filtering on.
    - lower_quantile (float): Lower quantile threshold (default is 0.25).
    - upper_quantile (float): Upper quantile threshold (default is 0.75).

    Returns:
    - pd.DataFrame: The filtered DataFrame.
    """
    # Initialize mask as True for all rows
    mask = pd.Series([True] * len(df))
    
    for col in columns:
        lower = df[col].quantile(lower_quantile)
        upper = df[col].quantile(upper_quantile)
        # Update mask to keep rows within the quantile range for each column
        mask &= df[col].between(lower, upper)
        print(f"Filtering '{col}': between {lower} and {upper}")
    
    # Apply the mask to the DataFrame
    filtered_df = df[mask].reset_index(drop=True)
    return filtered_df

def top_k_normalize(pi, k=10):
    # pi is of shape (m, n)
    # For each row, keep top k entries, zero out the rest, then normalize.

    m, n = pi.shape
    pi_normalized = np.zeros_like(pi)

    for i in range(m):
        row = pi[i, :]
        # Find the indices of the top k values in this row
        # argsort returns ascending order, so we take the last k for top k
        if k < n:
            top_indices = np.argsort(row)[-k:]
        else:
            top_indices = np.arange(n)  # If k >= n, keep all

        # Create a mask for the top k values
        mask = np.zeros_like(row, dtype=bool)
        mask[top_indices] = True

        # Zero out values not in top k
        row_topk = np.where(mask, row, 0.0)

        # Normalize so the sum of the row is 1
        row_sum = row_topk.sum()
        if row_sum > 0:
            row_topk /= row_sum

        pi_normalized[i, :] = row_topk

    return pi_normalized

def sample_from_transport_plan(X, Y, pi):
    """
    Given:
        X: numpy array of shape (m, d), source points
        Y: DataFrame or numpy array of shape (n, d), target points
        pi: numpy array of shape (m, n), transport plan
           - Each row of pi sums to 1, representing a probability distribution over Y for that row of X.
           
    Returns:
        samples: numpy array of shape (m, d), where each row is drawn from Y 
                 according to the probabilities in the corresponding row of pi.
    """
    import numpy as np

    m = X.shape[0]
    n = Y.shape[0]
    
    # If Y is a DataFrame, use .iloc or convert to numpy array
    # If Y is already a numpy array, this step is not needed
    if hasattr(Y, 'iloc'):
        # Y is a DataFrame
        get_row = lambda idx: Y.iloc[idx].values
    else:
        # Y is a numpy array
        get_row = lambda idx: Y[idx]

    samples = np.zeros((m, Y.shape[1]))
    pi = pi / pi.sum(axis=1, keepdims=True)  # ensure each row sums to 1

    for i in range(m):
        chosen_index = np.random.choice(np.arange(n), p=pi[i])
        samples[i] = get_row(chosen_index)
    
    df = pd.DataFrame(samples, columns=Y.columns)    
    return df

def sample_from_transport_plan2(X, Y, pi):
    """
    Given:
        X: numpy array of shape (m, d), source points
        Y: DataFrame or numpy array of shape (n, d), target points
        pi: numpy array of shape (m, n), transport plan
           - Each row of pi sums to 1, representing a probability distribution over Y for that row of X.
           
    Returns:
        samples: DataFrame of shape (m, d), where each row is drawn from Y 
                 according to the probabilities in the corresponding row of pi.
    """
    import numpy as np
    import pandas as pd

    m = X.shape[0]
    n = Y.shape[0]
    
    # If Y is a DataFrame, set a method to get row values and remember column names
    if hasattr(Y, 'iloc'):
        get_row = lambda idx: Y.iloc[idx].values
        columns = Y.columns
    else:
        # Y is assumed to be a numpy array
        get_row = lambda idx: Y[idx]
        columns = [f"col_{i}" for i in range(Y.shape[1])]

    # Handle rows of pi that sum to zero by assigning a uniform distribution
    row_sums = pi.sum(axis=1)  # shape (m,)
    zero_sum_rows = (row_sums == 0)
    if zero_sum_rows.any():
        pi[zero_sum_rows] = 1.0 / n  # Assign uniform distribution for those rows

    # Normalize pi to ensure each row sums to 1
    pi = pi / pi.sum(axis=1, keepdims=True)
    
    # Check and handle NaNs if they still exist
    # Replace NaNs with 0 and re-normalize if needed
    if np.isnan(pi).any():
        pi = np.nan_to_num(pi, nan=0.0)
        row_sums = pi.sum(axis=1)
        zero_sum_rows = (row_sums == 0)
        if zero_sum_rows.any():
            pi[zero_sum_rows] = 1.0 / n
        pi = pi / pi.sum(axis=1, keepdims=True)

    # Now sample from the distributions
    samples = np.zeros((m, Y.shape[1]))
    for i in range(m):
        chosen_index = np.random.choice(np.arange(n), p=pi[i])
        samples[i] = get_row(chosen_index)

    df = pd.DataFrame(samples, columns=columns)    
    return df
