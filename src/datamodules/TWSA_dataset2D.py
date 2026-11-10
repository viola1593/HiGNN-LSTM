
import numpy as np
import torch
import warnings
import xarray as xr

from torch.utils.data import Dataset


# create cell dataset
class TWSA2DDataset(Dataset):
    def __init__(self, ds_local, time_period, target='twsa', delay=12, input_seq_len=12, output_seq_len=1, time_step=None,
                   input_vars=None, scaling_mean_std=None, features_nanfill=0., twsa_climatology=None,
                   era5_climatology=None, **kwargs):
        """
        ds_local: xarray dataset with coords (latitude, longitude, time)
        input_seq_len: number of time steps in the feature sequence length
        delay: number of time steps between end of sequence and target
        time_step: provide a specific time step to only get the data for that time step for prediction (default None)
        twsa_climatology: training climatology [month, lat, lon] in cm. If given, twsa is normalized
            as the anomaly twsa - C[month] and each sample carries C of its target months as twsa_offset
        era5_climatology: training climatology with one [month, lat, lon] variable per ERA5 input.
            If given, each of its variables is normalized as the anomaly var - C[var][month]"""
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

        self.twsa_offset = None
        if twsa_climatology is not None:
            # anomaly space: the per-pixel normalization below then scales twsa - C[month]
            clim = twsa_climatology.sel(month=self.ds_timeslice.time.dt.month).drop_vars('month')
            self.ds_timeslice['twsa'] = self.ds_timeslice['twsa'] - clim
            self.twsa_offset = torch.from_numpy(clim.values) # [time, lat, lon]

        if era5_climatology is not None:
            for var in era5_climatology.data_vars:
                self.ds_timeslice[var] = (self.ds_timeslice[var]
                                          - era5_climatology[var].sel(month=self.ds_timeslice.time.dt.month).drop_vars('month'))

        self.normalize_ds_timeslice()
        self.month_to_features()
        self.samples = self.get_sample_indices()
        
        self.features = torch.Tensor(np.stack([self.ds_timeslice[var].to_numpy() for var in self.input_vars], axis=1))
        self.target_id = self.input_vars.index(self.target)

        


    def get_sample_indices(self):
        feature_indices_list = []

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
            # keep only windows over consecutive months (GRACE has missing months)
            months = self.ds_timeslice.time.values.astype('datetime64[M]').astype(int)
            window = self.input_seq_len + self.delay + self.pred_len
            feature_indices_list = [t for t in feature_indices_list
                                    if months[t + self.delay + self.pred_len - 1] - months[t - self.input_seq_len] == window - 1]

        return feature_indices_list


    def normalize_ds_timeslice(self):
        mask_land =  self.land_ocean_mask # 1 is land and 0 is ocean
        mask_ocean =  ~mask_land
        mask_not_antarctica = (self.ds_timeslice.lsm.where(self.ds_timeslice['lat']<-60)*0).fillna(1).astype('bool').values
           
        if self.scaling_params is not None:
            # NaNs are filled after normalization and masking (see below), the same order as in
            # the training branch, so that a NaN inside the valid mask is encoded identically
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

                    # clamp the divisor: a pixel with a std below the floor is attenuated
                    # (its normalized std becomes std/epsilon < 1), never amplified
                    epsilon = float(ds_std.median() * 0.01)
                    ds_std = ds_std.clip(min=epsilon)

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
        input_features = self.features[t-self.input_seq_len:t] # shape [<input_seq_len, num_vars, h, w>]
        # get target
        pred_start = t + self.delay
        pred_end = t + self.delay + self.pred_len
        target = self.features[pred_start:pred_end][:, self.target_id] 
        mask = self.land_ocean_mask[t] & self.mask_not_antarctica[t] # mask for land and not antarctica
        
        sample = {
            'x_sequence': input_features,
            'y_timestep': target,
            'land_mask': torch.Tensor(mask)
        }
        if self.twsa_offset is not None:
            sample['twsa_offset'] = self.twsa_offset[pred_start:pred_end] # C of the target months, added back in predict_step
        return sample


