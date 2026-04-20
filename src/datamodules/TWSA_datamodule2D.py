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



def read_config(config_path):
    import yaml
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def collate_fn(batch):
  return {
      'x_sequence': torch.stack([x['x_sequence'] for x in batch]),
      'y_timestep': torch.stack([x['y_timestep'] for x in batch]),
      'land_mask': torch.stack([x['land_mask'] for x in batch]),
}

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
            **kwargs       
            ):
        super(TWSA_Datamodule2D, self).__init__()
        self.ds_path = ds_path
        self.input_seq_len = input_seq_len
        self.delay = delay
        self.target = target
        self.target_seq_len = output_seq_len
        self.train_val_test_split = train_val_test_split


        self.save_hyperparameters(logger=False)

        self.ds = xr.open_dataset(ds_path)
        if 'lat' not in self.ds.coords.keys():
            self.ds = self.ds.rename({"latitude":"lat" , "longitude":"lon"}) # make sure naming conventions are consistent
        if input_vars is not None:
            # subset the dataset to only include input variables
            self.ds = self.ds[input_vars] 
        self.input_vars = list(self.ds.data_vars.keys())

        self.train_val_test_periods = self.get_train_val_test_periods()
        self.static_features = self.get_static_feat() #needed for certain edge constructors

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
                                        input_vars=list(self.ds.data_vars.keys())
                                        )
        
        return DataLoader(dataset=self.data_pred,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False,
                    drop_last=False,
            collate_fn=collate_fn)





    
