import torch 
import numpy as np
def KGE(y_pred, y_true, land_sea_mask=None):
   '''Kling-Gupta Efficiency adapted from Hallouin, T. (XXXX). HydroEval: Streamflow Simulations Evaluator (Version 0.1.0). Zenodo. https://doi.org/10.5281/zenodo.2591217
   Adapted to work with PyTorch tensors of the shape (time_steps, n_pixels)
   KGE is calculated for every pixel separately'''

   """Original Kling-Gupta Efficiency (KGE) and its three components
   (r, α, β) as per `Gupta et al., 2009
   <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.

   Note, all four values KGE, r, α, β are returned, in this order.

   :Calculation Details:
      .. math::
         E_{\\text{KGE}} = 1 - \\sqrt{[r - 1]^2 + [\\alpha - 1]^2
         + [\\beta - 1]^2}
      .. math::
         r = \\frac{\\text{cov}(e, s)}{\\sigma({e}) \\cdot \\sigma(s)}
      .. math::
         \\alpha = \\frac{\\sigma(s)}{\\sigma(e)}
      .. math::
         \\beta = \\frac{\\mu(s)}{\\mu(e)}

      where *e* is the *evaluation* series, *s* is (one of) the
      *simulations* series, *cov* is the covariance, *σ* is the
      standard deviation, and *μ* is the arithmetic mean.

   :Parameters:
      - **y_pred** (*torch.Tensor*): The predicted values, shape (time_steps, n_pixels).
      - **y_true** (*torch.Tensor*): The true values, shape (time_steps, n_pixels).
      - **land_sea_mask** (*torch.Tensor*, optional): A mask tensor indicating land pixels, shape (n_pixels,). If provided, KGE will
        be calculated only for land pixels. Default is None.

   """
   
   # calculate error in timing and dynamics r
   # (Pearson's correlation coefficient)
   axis=0
   pred_mean = torch.mean(y_pred, axis=0) 
   true_mean = torch.mean(y_true, axis=0)
   true_std = torch.std(y_true, axis=0)
   pred_std = torch.std(y_pred, axis=0)

   r_num = torch.sum((y_pred - pred_mean) * (y_true - true_mean),
                  axis=axis)
   r_den = torch.sqrt(torch.sum((y_pred - pred_mean) ** 2,
                        axis=axis)
                  * torch.sum((y_true - true_mean) ** 2, axis=axis))
   p_corr = r_num / r_den
   p_corr =  torch.where(r_den==0, torch.nan, p_corr)
   # calculate error in spread of flow alpha
#  import pdb; pdb.set_trace()
   alpha = pred_std / true_std
   # make sure that we get nans (instead of infs) where ever we divide by 0 assuming those are the sea pixels
   alpha =  torch.where(true_std==0, torch.nan, alpha)
   
   
   # calculate error in volume beta (bias of mean discharge)
   beta = pred_mean / true_mean
   beta = torch.where(true_mean==0, torch.nan, beta)
   

   if land_sea_mask is not None:
      land_sea_mask=land_sea_mask.to(y_pred.device)
      land_sea_mask= land_sea_mask.bool()
      alpha =alpha[land_sea_mask] # only consider pixels with land, for the others the std will be 0 and division by 0 will result in nan or inf
      beta = beta[land_sea_mask]
      p_corr = p_corr[land_sea_mask]

   # calculate the Kling-Gupta Efficiency KGE

   kge_ = 1 - torch.sqrt((p_corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
   

   return torch.vstack((kge_, p_corr, alpha, beta))


def KGE_for_multistep_pred(y_pred, y_true, land_sea_mask=None):
   '''Kling-Gupta Efficiency adapted from Hallouin, T. (XXXX). HydroEval: Streamflow Simulations Evaluator (Version 0.1.0). Zenodo. https://doi.org/10.5281/zenodo.2591217
   Adapted to work with PyTorch tensors of the shape (time_steps, n_pixels)
   KGE is calculated for every pixel separately'''

   """Original Kling-Gupta Efficiency (KGE) and its three components
   (r, α, β) as per `Gupta et al., 2009
   <https://doi.org/10.1016/j.jhydrol.2009.08.003>`_.

   Note, all four values KGE, r, α, β are returned, in this order.

   :Calculation Details:
      .. math::
         E_{\\text{KGE}} = 1 - \\sqrt{[r - 1]^2 + [\\alpha - 1]^2
         + [\\beta - 1]^2}
      .. math::
         r = \\frac{\\text{cov}(e, s)}{\\sigma({e}) \\cdot \\sigma(s)}
      .. math::
         \\alpha = \\frac{\\sigma(s)}{\\sigma(e)}
      .. math::
         \\beta = \\frac{\\mu(s)}{\\mu(e)}

      where *e* is the *evaluation* series, *s* is (one of) the
      *simulations* series, *cov* is the covariance, *σ* is the
      standard deviation, and *μ* is the arithmetic mean.

   """
   
   # calculate error in timing and dynamics r
   # (Pearson's correlation coefficient)

      # if we predict a single value, the KGE is calculated for the entire validation time period
   kge_list = []
   alpha_list = []
   beta_list = []
   p_corr_list = []
   for seq, true in zip(y_pred, y_true):
      axis=0

      pred_mean = torch.mean(seq, axis=0)
      true_mean = torch.mean(true, axis=0)
      true_std = torch.std(true, axis=0)
      pred_std = torch.std(seq, axis=0)

      r_num = torch.sum((seq - pred_mean) * (true - true_mean),
                     axis=axis)
      r_den = torch.sqrt(torch.sum((seq - pred_mean) ** 2,
                           axis=axis)
                     * torch.sum((true - true_mean) ** 2, axis=axis))
      p_corr = r_num / r_den
      p_corr =  torch.where(r_den==0, torch.nan, p_corr)
      # calculate error in spread of flow alpha
   #  import pdb; pdb.set_trace()
      alpha = pred_std / true_std
      # make sure that we get nans (instead of infs) where ever we divide by 0 assuming those are the sea pixels
      alpha =  torch.where(true_std==0, torch.nan, alpha)
      
      
      # calculate error in volume beta (bias of mean discharge)
      beta = (torch.sum(seq, axis=axis)
            /(torch.sum(true, axis=axis))+1e-10)
      
      if land_sea_mask is not None:
         land_sea_mask=land_sea_mask.to(seq.device)
         land_sea_mask= land_sea_mask.bool()
         alpha = alpha[land_sea_mask] # only consider pixels with land, for the others the std will be 0 and division by 0 will result in nan or inf
         beta = beta[land_sea_mask]
         p_corr = p_corr[land_sea_mask]

      # calculate the Kling-Gupta Efficiency KGE

      kge_ = 1 - torch.sqrt((p_corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
      kge_list.append(kge_)
      alpha_list.append(alpha)
      beta_list.append(beta)
      p_corr_list.append(p_corr)
   

   return torch.vstack((kge_list, p_corr_list, alpha_list, beta_list))