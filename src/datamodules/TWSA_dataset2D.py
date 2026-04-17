
import xarray as xr
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch

import random
import time
from tqdm import tqdm
import warnings

def datetime2float(time_values):
    return (time_values - np.datetime64('1979-07-01T00:00:00')) / np.timedelta64(1, 'M')




        

# create cell dataset
class TWSA2DDataset(Dataset):
    def __init__(self, ds_local, time_period, target='twsa', delay=12, input_seq_len=12, output_seq_len=1, time_step=None,
                   input_vars=None, scaling_mean_std=None, features_nanfill=0., **kwargs):
        """
        ds_local: xarray dataset with coords (latitude, longitude, time)
        input_seq_len: number of time steps in the feature sequence length
        delay: number of time steps between end of sequence and target
        time_step: provide a specific time step to only get the data for that time step for prediction (default None)"""
        self.ds_local = ds_local
    
        self.ds_timeslice = ds_local.sel(time=slice(time_period[0], time_period[1])).astype(np.float32)
       ## print data statistics
        #print(self.ds_timeslice.to_dataframe().describe())
        
        if not input_vars:
            input_vars = list(self.ds_timeslice.data_vars.keys())
        self.input_vars = input_vars
        self.scaling_params = scaling_mean_std
        
        self.input_seq_len = input_seq_len
        self.target = target
        self.delay = delay
        self.pred_len = output_seq_len
        
        
        self.features_nanfill = features_nanfill
        self.land_threshold = 0.8
        self.land_ocean_mask = self.ds_timeslice['lsm'].values>self.land_threshold
        self.land_variables = [ 'ro', 'pev', 'swvl1', 'swvl2', 'swvl3', 'swvl4','lai_hv', 'lai_lv']
        self.ocean_variables = ['sst']
        self.mask_not_antarctica = (self.ds_timeslice.lsm.where(self.ds_timeslice['lat']<-60)*0).fillna(1).astype('bool').values
        
        self.time_step = time_step
        
        self.normalize_ds_timeslice()
        self.month_to_features()
        self.samples = self.get_sample_indices()
        
        self.features = torch.Tensor(np.stack([self.ds_timeslice[var].to_numpy() for var in self.input_vars], axis=1))
        self.target_id = self.input_vars.index(self.target)
        #self.df = self.ds_timeslice.to_dataframe().reset_index().set_index(['time'])
        # self.features, self.targets, self.positions = self.prepare_samples()
        
        # print(self.features.shape)

    def get_sample_indices(self):
        feature_indices_list = []
        #import pdb; pdb.set_trace()

        if self.time_step is not None:
            # this is for inference when we only want to predict one time step
            t = self.time_step
            if t < self.input_seq_len:
                print(f"Time step {t} is before the local lag {self.input_seq_len}, setting time step to {self.input_seq_len}")
                t += self.input_seq_len
            if t > len(self.ds_timeslice.time) - self.delay-self.pred_len:
                print(f"Time step {t} is after the last possible time step, setting time step to {len(self.ds_timeslice.time) - self.delay}")
                t = len(self.ds_timeslice.time) - self.delay-self.pred_len
            feature_indices_list = [t]
                
      
        else: 
            feature_indices_list = [t for t in range(self.input_seq_len, len(self.ds_timeslice.time)-self.delay-self.pred_len+1)]
                        
        return feature_indices_list


    def normalize_ds_timeslice_1D(self):       
        # normalize input variables 
        mask_land =  self.land_ocean_mask # 1 is land and 0 is ocean
        mask_ocean =  self.ds_timeslice['lsm'].values==0
           
        if self.scaling_params is not None:
            self.ds_timeslice = self.ds_timeslice.fillna(self.features_nanfill)
            for i, var in enumerate(self.input_vars):
                assert var == self.scaling_params[i][0].name
                assert var == self.scaling_params[i][1].name

                self.ds_timeslice[var] = (self.ds_timeslice[var] - self.scaling_params[i][0]) / self.scaling_params[i][1]
                
                if var in self.land_variables:
                    mask = mask_land
                elif var in self.ocean_variables:
                    mask = mask_ocean
                elif var== 'twsa':
                    mask = (self.mask_not_antarctica & mask_land)
                else:
                    mask = np.ones_like(mask_land).astype('bool')
                self.ds_timeslice[var].values[~mask] = 0 #self.ds_timeslice[var].where(~mask, 0).fillna(self.features_nanfill)                
                self.ds_timeslice[var] = self.ds_timeslice[var].fillna(self.features_nanfill)
   
    def normalize_ds_timeslice(self):
        mask_land =  self.land_ocean_mask # 1 is land and 0 is ocean
        mask_ocean =  self.ds_timeslice['lsm'].values==0
        mask_not_antarctica = (self.ds_timeslice.lsm.where(self.ds_timeslice['lat']<-60)*0).fillna(1).astype('bool').values
           
        if self.scaling_params is not None:
            self.ds_timeslice = self.ds_timeslice.fillna(self.features_nanfill)
            for i, var in enumerate(self.input_vars):
                try:
                    assert var == self.scaling_params[i][0].name
                    assert var == self.scaling_params[i][1].name
                
                    self.ds_timeslice[var] = (self.ds_timeslice[var] - self.scaling_params[i][0]) / self.scaling_params[i][1]
                except (AssertionError, AttributeError): print(f"Variable {var} not found in scaling parameters, skipping normalization")
                
                if var in self.land_variables:
                    mask = mask_land
                elif var in self.ocean_variables:
                    mask = mask_ocean
                elif var== 'twsa':
                    mask = (mask_not_antarctica & mask_land)
                else:
                    mask = np.ones_like(mask_land).astype('bool')
                self.ds_timeslice[var].values[~mask] = 0 #self.ds_timeslice[var].where(~mask, 0).fillna(self.features_nanfill)
                self.ds_timeslice[var] = self.ds_timeslice[var].fillna(self.features_nanfill)
          
        else:
            epsilon=1e-6
            
            scaling_mean_std =[]
            for var in self.input_vars:
                if var in self.land_variables:
                    mask = mask_land
                elif var in self.ocean_variables:
                    mask = mask_ocean
                elif var== 'twsa':
                    mask = (self.mask_not_antarctica & mask_land)
                elif var == 'lsm':
                    scaling_mean_std.append((0, 1))
                    continue # do not normalize land sea mask
                else:
                    mask = np.ones_like(mask_land).astype('bool')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ds_mean = self.ds_timeslice[var].where(mask).mean(dim='time')
                    ds_std = self.ds_timeslice[var].where(mask).std(dim='time')
                    ds_std = ds_std.where(ds_std.values>=epsilon, other=epsilon) # make sure not too small otherwise numerical instability will happen

                self.ds_timeslice[var] = (self.ds_timeslice[var]- ds_mean) / ds_std
                self.ds_timeslice[var].values[~mask] = 0 #self.ds_timeslice[var].where(~mask, 0).fillna(self.features_nanfill)
                self.ds_timeslice[var] = self.ds_timeslice[var].fillna(self.features_nanfill)
                scaling_mean_std.append((ds_mean, ds_std))
            self.scaling_params = scaling_mean_std 
   
    
    def month_to_features(self):
        time_vals = self.ds_timeslice.time.values
        months = (time_vals.astype('datetime64[M]').astype(int) % 12)
            # Create a new variable with proper dimensions (time, lat, lon)
        cos_enc = np.cos(2 * np.pi * months / 12)
        sin_enc = np.sin(2 * np.pi * months / 12)
        cos_enc_expanded = np.broadcast_to(
            cos_enc[:, np.newaxis, np.newaxis],
            (len(time_vals), len(self.ds_timeslice.lat), len(self.ds_timeslice.lon)))
        sin_enc_expanded = np.broadcast_to(
            sin_enc[:, np.newaxis, np.newaxis],
            (len(time_vals), len(self.ds_timeslice.lat), len(self.ds_timeslice.lon)))

        # Add as new variable to dataset with proper coordinates
        self.ds_timeslice['cos_time'] = xr.DataArray(
            cos_enc_expanded,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': self.ds_timeslice.time,
                'lat': self.ds_timeslice.lat,
                'lon': self.ds_timeslice.lon})
                # Add as new variable to dataset with proper coordinates
        self.ds_timeslice['sine_time'] = xr.DataArray(
            sin_enc_expanded,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': self.ds_timeslice.time,
                'lat': self.ds_timeslice.lat,
                'lon': self.ds_timeslice.lon})
        self.input_vars += ['cos_time', 'sine_time']
        
    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        # get sample
        t = self.samples[idx]
        # get local input
        #timeslice_input_ds = self.ds_timeslice.isel(time=slice(t-self.input_seq_len, t))

        input_features = self.features[t-self.input_seq_len:t] # shape [<input_seq_len, num_vars, h, w>]
        #local_input = np.stack([timeslice_input_ds[var].to_numpy() for var in self.input_vars], axis=1)
        
        
        # local_pos = np.stack([local_input_ds[var].values for var in self.positional_vars], axis=0)
        #local_input = np.nan_to_num(local_input, nan=self.features_nanfill)
        # get target
        pred_start = t + self.delay
        pred_end = t + self.delay + self.pred_len
        target = self.features[pred_start:pred_end][:, self.target_id] 
        # local_target = self.ds_timeslice.isel(time=slice(pred_start, pred_end))[self.target].values

        # local_target = np.squeeze(local_target)
        # mask = self.land_mask[pred_start:pred_end] & self.mask_not_antarctica[pred_start:pred_end] # mask for land and not antarctica
        
        mask = self.land_ocean_mask[t] & self.mask_not_antarctica[t] # mask for land and not antarctica
        
        return {
            # x_sequence has shape (1, timesteps, features, h, w)
            'x_sequence': input_features,            
            # y_local has shape (1, timesteps, h, w)
            'y_timestep': target,
            'land_mask': torch.Tensor(mask)
        }



if __name__ == '__main__':
    path= "data/twsa_rec_era5_global_1deg_detrend.nc" #SeasFireCube_v3_1deg
    ds=xr.open_dataset(path)
    # ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    print("data loaded")
    input_vars =   ['t2m_mean','tp','vpd']
    log_transform_vars = []
    total_time_steps = len(ds.time)
    train_time_steps = int(total_time_steps * 0.6)
    val_time_steps = int(total_time_steps * 0.2)
    time_period= (ds.time[0].values, ds.time[3].values)
    
    seq_len = 1
    output_seq_len = 1
    dataset = TWSA2DDataset(ds, 
                            time_period, 
                            target='twsa', 
                            delay=12, 
                            input_seq_len=12, 
                            time_step=None,
                            )
    import matplotlib.pyplot as plt
    import pdb; pdb.set_trace()
    dataset.ds_timeslice.twsa.max(dim=['lon', 'lat']).plot(label='max')
    dataset.ds_timeslice.twsa.min(dim=['lon', 'lat']).plot(label='min')
    plt.legend()
    plt.savefig('twsa_max_min_detrend_norm.png')
    plt.close()