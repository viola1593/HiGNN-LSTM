import numpy as np
import torch

def mse_with_nans(y_true, y_pred):
    # Mask to ignore NaN values
    mask = ~torch.isnan(y_true) & ~torch.isnan(y_pred)
    
    # Calculate MSE only for non-NaN values
    mse = np.mean((y_true[mask] - y_pred[mask]) ** 2)
    
    return mse

def mean_with_nans(x):
    mask = ~torch.isnan(x)
    return torch.mean(x[mask])

def rmse_with_nans(y_true, y_pred):
    return np.sqrt(mse_with_nans(y_true, y_pred))

def mean_abs_with_nans(x):
    mask = ~torch.isnan(x) 
    return torch.mean(torch.abs(x[mask]))

def median_with_nans(x):
    mask = ~torch.isnan(x)
    return torch.median(x[mask])

