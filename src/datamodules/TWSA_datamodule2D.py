from typing import Optional, Tuple, Dict
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import  DataLoader, Dataset

import numpy as np
import xarray as xr

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datamodules.TWSA_dataset2D import TWSA2DDataset

GRACE_PATH = 'data/processed/GRACE_era5.nc'


def read_config(config_path):
    import yaml
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def collate_fn(batch):
  # stacks every key of the samples, so 'twsa_offset' only when the datasets are in anomaly mode
  return {key: torch.stack([x[key] for x in batch]) for key in batch[0]}

class TWSA_Datamodule2D(LightningDataModule):

    def __init__(
            self,
            ds_path: str = None,
            input_vars: list = None,
            input_seq_len: int = 12,
            delay: int = 12,
            target: str = 'twsa',
            output_seq_len: int = 1, 

            train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
            twsa_anomaly: bool = False, # models see and predict twsa - C_train[month]
            era5_anomaly: bool = False, # models see every input except twsa and lsm as var - C_train[month]
            **kwargs
            ):
        super(TWSA_Datamodule2D, self).__init__()
        self.ds_path = ds_path
        self.input_seq_len = input_seq_len
        self.delay = delay
        self.target = target
        self.target_seq_len = output_seq_len
        self.train_val_test_split = train_val_test_split
        self.twsa_anomaly = twsa_anomaly
        self.era5_anomaly = era5_anomaly

        self.save_hyperparameters(logger=False)

        self.ds = xr.open_dataset(ds_path)
        if 'lat' not in self.ds.coords.keys():
            self.ds = self.ds.rename({"latitude":"lat" , "longitude":"lon"}) # make sure naming conventions are consistent
        if input_vars is not None:
            # subset the dataset to only include input variables
            self.ds = self.ds[input_vars] 
        self.input_vars = list(self.ds.data_vars.keys())

        self.train_val_test_periods = self.get_train_val_test_periods()
        # raw training climatology [month, lat, lon] in cm, also needed by eval in both modes.
        # float32 so it does not promote the datasets' twsa, NaN (Antarctica, ocean) -> 0
        self.twsa_climatology = (self.ds['twsa'].sel(time=slice(*self.train_val_test_periods['train']))
                                 .groupby('time.month').mean('time').fillna(0).astype(np.float32))
        # same for every other input (the ERA5 variables), NaN outside a variable's domain -> 0.
        # not lsm: it is static, its climatology is itself and the anomaly would be 0.
        # only built when used: with input_vars=['twsa', 'lsm'] there is nothing to average
        self.era5_climatology = (self.ds[[v for v in self.input_vars if v not in ('twsa', 'lsm')]]
                                 .sel(time=slice(*self.train_val_test_periods['train']))
                                 .groupby('time.month').mean('time').fillna(0).astype(np.float32)
                                 if era5_anomaly else None)

        self.scaling_mean_std = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def get_train_val_test_periods(self):     
        self.total_time_steps = len(self.ds.time)
        train_time_steps = int(self.total_time_steps * self.train_val_test_split[0])
        val_time_steps = int(self.total_time_steps * self.train_val_test_split[1])

        train_data_start = self.ds.time[0].values
        train_data_end = self.ds.time[train_time_steps].values
        val_data_start = self.ds.time[train_time_steps+1].values
        
        val_data_end = self.ds.time[train_time_steps + val_time_steps].values
        test_data_start = self.ds.time[train_time_steps + val_time_steps+1].values
        test_data_end = self.ds.time[-1].values
        return {'train': (train_data_start, train_data_end),
                'val': (val_data_start, val_data_end),
                'test': (test_data_start, test_data_end)}

    def get_static_feat(self, static_var=None):
        if static_var is None:
            # Get the values of the coordinates
            latitude_values = torch.Tensor(self.ds.coords['lat'].values)
            longitude_values = torch.Tensor(self.ds.coords['lon'].values)
              # Create a 2D tensor from lat and lon values
            lat_grid, lon_grid = torch.meshgrid(torch.tensor(latitude_values), torch.tensor(longitude_values))
            static_feat = torch.stack([lon_grid, lat_grid ], dim=-1).reshape(-1, 2)  
        else:  
            static_feat = torch.stack([torch.Tensor(self.ds.isel(time=0)[var].values.reshape(-1)) for var in static_var], dim=-1)
        return static_feat

          

    def setup(self, stage: Optional[str] = None):
        """ Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        
        if stage == "fit":
            # load datasets only if they're not loaded already
            if not self.data_train:                
                self.data_train = TWSA2DDataset(self.ds.copy(), time_period=self.train_val_test_periods['train'],
                                                    
                                                    target='twsa', 
                                                    delay=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    
                                                    twsa_climatology=self.twsa_climatology if self.twsa_anomaly else None,
                                                    era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                                    input_vars=list(self.ds.data_vars.keys()), 
                                                     
                                                    )
                
                self.scaling_mean_std = self.data_train.scaling_params

        
        
            # load datasets only if they're not loaded already
            if not self.data_val:
                print("Scaling data for prediction with scaling mean std derived from training data")
                self.data_val = TWSA2DDataset(self.ds.copy(), time_period=self.train_val_test_periods['val'],
                                       
                                                    target='twsa', 
                                                    delay=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    twsa_climatology=self.twsa_climatology if self.twsa_anomaly else None,
                                                    era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                                    input_vars=list(self.ds.data_vars.keys()), 
                                                     
                                                    )
        if stage == "validate":
                    # load datasets only if they're not loaded already
            if not self.data_val:
                if self.scaling_mean_std is None:
                    print("Mean and std not found, scaling data for validation with scaling mean std derived from validation data")
                self.data_val = TWSA2DDataset(self.ds.copy(), time_period=self.train_val_test_periods['val'],
                               
                                                    target='twsa', 
                                                    delay=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    twsa_climatology=self.twsa_climatology if self.twsa_anomaly else None,
                                                    era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                                    input_vars=list(self.ds.data_vars.keys()), 
                                                      
                                                    )
                
  
        if stage == "test":
                # load datasets only if they're not loaded already
            if not self.data_test:
                if self.scaling_mean_std is None:
                    print("Mean and std not found, scaling data for test with scaling mean std derived from test data")
                self.data_test = TWSA2DDataset(self.ds.copy(), time_period=self.train_val_test_periods['test'],
                          
                                                    target='twsa', 
                                                    delay=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    twsa_climatology=self.twsa_climatology if self.twsa_anomaly else None,
                                                    era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                                    input_vars=list(self.ds.data_vars.keys()), 
                                                     
                                                   )
        if stage == "predict":
            pass

           


    def unnormalize_twsa(self, data):
        """Unnormalize the TWSA data using the mean and std from the training data.
        data shapes: (batch_size*num_nodes, num_timesteps)
        or data is an xr.DataArray
        """
      
        assert self.scaling_mean_std is not None, "Mean and std not found"
        assert self.scaling_mean_std[0][0].name == 'twsa', "Mean and std not found for twsa"

        if isinstance(data, torch.Tensor):
            std = torch.Tensor(self.scaling_mean_std[0][1].values).to(data.device)
            mean = torch.Tensor(self.scaling_mean_std[0][0].values).to(data.device)
        else: 
            std = self.scaling_mean_std[0][1].values
            mean = self.scaling_mean_std[0][0].values
        rescaled = data * std + mean
            
        return rescaled 
    
    def get_landmask_without_antarctica(self):
        """True for any land outside antarctica"""
        if self.data_train is not None:
            return (self.data_train.mask_not_antarctica[0] & self.get_landmask())
        elif self.data_val is not None:
            return (self.data_val.mask_not_antarctica[0] & self.get_landmask())
        elif self.data_test is not None:
            return (self.data_test.mask_not_antarctica[0] & self.get_landmask())
        else:
            return None
        
    def get_landmask(self):
        if self.data_train is not None:
            return self.data_train.land_ocean_mask[0]
        elif self.data_val is not None:
            return self.data_val.land_ocean_mask[0]
        elif self.data_test is not None:
            return self.data_test.land_ocean_mask[0]
        else:
            return None

    def train_dataloader(self):
        
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn
        )
    def predict_dataloader(self, time_step_idx: int=None, period=None):
        if period is None:
            period = self.train_val_test_periods['val']
            print("No time period specified. Predicting on validation period")
        if self.scaling_mean_std is None:
            print("Mean and std not found, scaling data for prediction with scaling mean std derived from prediction dataset")
        self.data_pred = TWSA2DDataset(self.ds.copy(), 
                                        time_period=period, 
                                        time_step=time_step_idx,
                                        target='twsa', 
                                        delay=self.delay, 
                                        input_seq_len=self.input_seq_len,
                                        output_seq_len=self.target_seq_len,
                                        scaling_mean_std=self.scaling_mean_std,
                                        twsa_climatology=self.twsa_climatology if self.twsa_anomaly else None,
                                        era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                        input_vars=list(self.ds.data_vars.keys())
                                        )
        
        return DataLoader(dataset=self.data_pred,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False,
                    drop_last=False,
            collate_fn=collate_fn)

    def GRACE_dataloader(self, time_step_idx: int=None, twsa_climatology=None):
        """Predict on GRACE TWSA (same grid as the reconstruction), scaled with the training data.
        twsa_climatology: [month, lat, lon] used instead of the training climatology for the twsa
            input anomaly and twsa_offset (anomaly mode only). ERA5 climatology and scaling stay the training ones."""
        if twsa_climatology is not None and not self.twsa_anomaly:
            raise ValueError("twsa_climatology needs twsa_anomaly=True")
        if twsa_climatology is None:
            twsa_climatology = self.twsa_climatology
        grace = xr.open_dataset(GRACE_PATH)
        grace['twsa'] = xr.where(grace['lsm'], grace['twsa'], 0.0)  # zero ocean, as in the HiGNN GRACE loader
        grace = grace[self.input_vars]  # drop extra variables, training order (scaling params match by position)
        self.data_pred = TWSA2DDataset(grace,
                                        time_period=(grace.time[0].values, grace.time[-1].values),
                                        time_step=time_step_idx,
                                        target='twsa',
                                        delay=self.delay,
                                        input_seq_len=self.input_seq_len,
                                        output_seq_len=self.target_seq_len,
                                        scaling_mean_std=self.scaling_mean_std,
                                        twsa_climatology=twsa_climatology if self.twsa_anomaly else None,
                                        era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                        input_vars=list(self.input_vars)
                                        )

        return DataLoader(dataset=self.data_pred,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False,
                    drop_last=False,
            collate_fn=collate_fn)





    
