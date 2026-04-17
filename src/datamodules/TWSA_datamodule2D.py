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

class TWSA_Datamodule2D_new(LightningDataModule):

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
        super(TWSA_Datamodule2D_new, self).__init__()

   
        self.ds_path = ds_path
        self.input_seq_len = input_seq_len
        self.delay = delay
        print("Delay", self.delay)
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

                

            # Save mean std dict for normalization during inference time
            #print("Mean", self.mean_std_dict)
            #with open(f'mean_std_dict_{self.target_shift}.json', 'w') as f:
            #    f.write(json.dumps(self.mean_std_dict))

        
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

           


    def unnormalize_twsa_1D(self, data):
        
        assert self.scaling_mean_std is not None, "Mean and std not found"
        assert self.scaling_mean_std[0][0].name == 'twsa', "Mean and std not found for twsa"
        if isinstance(data, torch.Tensor):
            rescaled = data * torch.Tensor(self.scaling_mean_std[0][1].values).to(data.device) + torch.Tensor(self.scaling_mean_std[0][0].values).to(data.device)
               
        else: 
            rescaled = data * self.scaling_mean_std[0][1].values + self.scaling_mean_std[0][0].values
        # set all non land pixels (including antarctica where twsa is not measured) to 0
        mask = (self.get_landmask_without_antarctica() & self.get_landmask())
        while mask.ndim < data.ndim:
            mask = mask[None, ...]  # or mask = np.expand_dims(mask, axis=0)
        if isinstance(data, xr.DataArray):
            rescaled = rescaled.where(mask, 0) # where mask is true return original value, 0 otherwise
        elif isinstance(rescaled, torch.Tensor):
            rescaled[~torch.Tensor(mask).expand_as(rescaled).bool()]=0
        else:
            rescaled[~mask] = 0
        return rescaled 




    
    def unnormalize_twsa(self, data):
        """Unnormalize the TWSA data using the mean and std from the training data.
        data shapes: (batch_size*num_nodes, num_timesteps)
        or data is an xr.DataArray
        """
        # num_nodes = self.get_num_nodes()
        assert self.scaling_mean_std is not None, "Mean and std not found"
        assert self.scaling_mean_std[0][0].name == 'twsa', "Mean and std not found for twsa"

        if isinstance(data, torch.Tensor):
            std = torch.Tensor(self.scaling_mean_std[0][1].values).to(data.device)
            mean = torch.Tensor(self.scaling_mean_std[0][0].values).to(data.device)
        else: 
            std = self.scaling_mean_std[0][1].values
            mean = self.scaling_mean_std[0][0].values
        rescaled = data * std + mean
        # if std.ndim == 1:  # simple normalization: mean/std over space and time         
        #     rescaled = data * std + mean
        # elif std.ndim == 2: # pixel-wise normalization: mean/std for each pixel over time  
            
        #     lsm = self.get_landmask_without_antarctica()     
        #     if data.ndim == 2: # [batch_size*num_nodes, num_timesteps]
                
        #         batch_size = data.shape[0]
        #         if isinstance(data, torch.Tensor):
        #             std = torch.concat((std[lsm], torch.zeros(self.num_ocean_clusters).to(data.device)), dim=0) # get into shape of nodes (land+ocean)
        #             mean = torch.concat((mean[lsm], torch.zeros(self.num_ocean_clusters).to(data.device)), dim=0)
            
        #             std = std.unsqueeze(1).repeat(batch_size, data.shape[-1]) # get into shape of timesteps
        #             mean = mean.unsqueeze(1).repeat(batch_size, data.shape[-1])
        #         else:
        #             std = np.concatenate((std[lsm], np.zeros(self.num_ocean_clusters)), axis=0)
        #             mean = np.concatenate((mean[lsm], np.zeros(self.num_ocean_clusters)), axis=0)
                    
        #             std = np.tile(std[:, None], (batch_size, data.shape[-1]))  
        #             mean = np.tile(mean[:, None], (batch_size, data.shape[-1]))  
        #     elif data.ndim == 1:   
        #         if isinstance(data, torch.Tensor):
        #             std = torch.concat((std[lsm], torch.zeros(self.num_ocean_clusters).to(data.device)), dim=0) # get into shape of nodes (land+ocean)
        #             mean = torch.concat((mean[lsm], torch.zeros(self.num_ocean_clusters).to(data.device)), dim=0)
        #         else:
        #             std = np.concatenate((std[lsm], np.zeros(self.num_ocean_clusters)), axis=0)
        #             mean = np.concatenate((mean[lsm], np.zeros(self.num_ocean_clusters)), axis=0) 
        #     elif data.ndim == 3:
        #         assert isinstance(data, xr.DataArray) , "3D data should be xr.DataArray for rescaling"
        #     rescaled = data * std + mean
            
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
        #print(self.hparams.batch_size)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            #pin_memory=self.hparams.pin_memory,
            shuffle=True,
            # persistent_workers=True,
            drop_last=False,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            #pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # persistent_workers=True,
            drop_last=False,
            collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            #pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # persistent_workers=True,
            drop_last=False,
            collate_fn=collate_fn
        )
    def predict_dataloader(self, time_step_idx: int=None, period=None):
        if period is None:
            period = self.train_val_test_periods['val']
            print("No time period specified. Predicting on validation period")
        if self.scaling_mean_std is None:
            print("Mean and std not found, scaling data for prediction with scaling mean std derived from prediction dataset")
        self.data_pred = TWSA2DDataset(self.ds.copy(), time_period=period, time_step=time_step_idx,
                                        
                                                    target='twsa', 
                                                    delay=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    input_vars=list(self.ds.data_vars.keys()), 
                                                      
                                                    )
        return DataLoader(dataset=self.data_pred,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    #pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    # persistent_workers=True,
                    drop_last=False,
            collate_fn=collate_fn)







# class TWSA_Datamodule2D(LightningDataModule):
#     """CAUTION: I dont know what's happening here, dont use!
    
    
#     Example of LightningDataModule for MNIST dataset.

#     A DataModule implements 5 key methods:
#         - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
#         - setup (things to do on every accelerator in distributed mode)
#         - train_dataloader (the training dataloader)
#         - val_dataloader (the validation dataloader(s))
#         - test_dataloader (the test dataloader(s))

#     This allows you to share a full dataset without explaining how to download,
#     split, transform and process the data.

#     Read the docs:
#         https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
#     """

#     def __init__(
#             self,
#             ds_path: str = None,
#             input_vars: list = None,
#             log_transform_vars: list = None,
#             positional_vars: list = None,
#             oci_vars: list = None,
#             local_lag: int = 12,
#             horizon: int = 12,
#             oci_lag: int = 10,
#             target: str = 'twsa',
#             target_shift: int = 1,
#             batch_size: int = 64,
#             num_workers: int = 8,
#             pin_memory: bool = False,
#             debug: bool = False,
#             aoi_longitude:  Tuple[float, float] = None, #(-13, 45),
#             aoi_latitude: Tuple[float, float] = None, # (70, 25),
#             train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
         
            
#     ):
#         super().__init__()

#         # this line allows to access init params with 'self.hparams' attribute
#         if positional_vars is None:
#             self.positional_vars = []
#         else:
#             self.positional_vars = positional_vars
#         self.ds_path = ds_path
#         self.input_vars = input_vars
#         self.log_transform_vars = log_transform_vars
#         self.oci_vars = oci_vars
#         self.local_lag = local_lag
#         self.horizon = horizon
#         self.oci_lag = oci_lag
#         self.target = target
#         self.target_shift = target_shift
#         self.train_val_test_split = train_val_test_split

#         self.mean_std_dict = None
#         if debug:
#             self.num_timesteps = 60
#         else:
#             self.num_timesteps = -1
#         if aoi_longitude:
#             self.region = {'longitude': slice(aoi_longitude[0],aoi_longitude[-1]),
#                         'latitude': slice(aoi_latitude[0], aoi_latitude[-1])}
#         else: self.region = None

#         self.save_hyperparameters(logger=False)

#         self.ds = xr.open_dataset(ds_path)
#         self.ds = self.ds.rename({"lat": "latitude", "lon": "longitude"})
#         if self.input_vars is None:
#             self.input_vars = list(self.ds.data_vars.keys())
#         # self.ds['sst'] = self.ds['sst'].where(self.ds['sst'] >= 0)

#         self.train_val_test_periods = self.get_train_val_test_periods()

        

        
#         self.scaling_mean_std = None
#         self.data_train: Optional[Dataset] = None
#         self.data_val: Optional[Dataset] = None
#         self.data_test: Optional[Dataset] = None


#     def get_train_val_test_periods(self):
          
#         self.total_time_steps = len(self.ds.time)
#         train_time_steps = int(self.total_time_steps * self.train_val_test_split[0])
#         val_time_steps = int(self.total_time_steps * self.train_val_test_split[1])

#         train_data_start = self.ds.time[0].values
#         train_data_end = self.ds.time[train_time_steps].values
#         val_data_start = self.ds.time[train_time_steps+1].values
#         val_data_end = self.ds.time[train_time_steps + val_time_steps].values
#         test_data_start = self.ds.time[train_time_steps + val_time_steps + 1].values
#         test_data_end = self.ds.time[-1].values
#         return {'train': (train_data_start, train_data_end),
#                 'val': (val_data_start, val_data_end),
#                 'test': (test_data_start, test_data_end)}
        

#     def setup(self, stage: Optional[str] = None):
#         """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

#         This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
#         so be careful not to execute the random split twice! The `stage` can be used to
#         differentiate whether it's called before trainer.fit()` or `trainer.test()`.
#         """
        
#         if stage == "fit":
#             # load datasets only if they're not loaded already
#             if not self.data_train:
#                 train_batches, train_oci_batches, scaling_mean_std = load_datasets(self.ds.copy(),
#                                                                     time_period=self.train_val_test_periods['train'],
#                                                                     input_vars=list(self.ds.data_vars.keys()),

#                                                                     num_timesteps=self.num_timesteps,
#                                                                     region=self.region)
                
#                 self.data_train = TWSA2DDataset(train_batches,
#                                                 target='twsa', 
#                                                 delay=self.horizon, 
#                                                 input_seq_len=self.local_lag,
#                                                 ds_oci=None, 
#                                                 oci_lag=None, 
#                                                 input_vars=None, 
#                                                 positional_vars=['longitude', 'latitude'], 
#                                                 oci_vars=None, 
#                                                 features_nanfill=-1.)
#                 self.scaling_mean_std = scaling_mean_std

#             # Save mean std dict for normalization during inference time
#             #print("Mean", self.mean_std_dict)
#             #with open(f'mean_std_dict_{self.target_shift}.json', 'w') as f:
#             #    f.write(json.dumps(self.mean_std_dict))

        
#             # load datasets only if they're not loaded already
#             if not self.data_val:
#                 val_batches, val_oci_batches, scaling_mean_std = load_datasets(self.ds.copy(),
#                                                                         time_period=self.train_val_test_periods['val'],
#                                                                         input_vars=list(self.ds.data_vars.keys()),
 
#                                                                         num_timesteps=self.num_timesteps,
#                                                                         region=self.region, 
#                                                                         scaling_mean_std=self.scaling_mean_std)
                    
#                 self.data_val = TWSA2DDataset(val_batches,
#                                                     target='twsa', 
#                                                 delay=self.horizon, 
#                                                 input_seq_len=self.local_lag,
#                                                 ds_oci=None, 
#                                                 oci_lag=None, 
#                                                 input_vars=None, 
#                                                 positional_vars=['longitude', 'latitude'], 
#                                                 oci_vars=None, 
#                                                 features_nanfill=-1.)
#         if stage == "validate":
#                     # load datasets only if they're not loaded already
#             if not self.data_val:
#                 print("Scaling data for prediction with scaling mean std derived from training data")
#                 val_batches, val_oci_batches, scaling_mean_std = load_datasets(self.ds.copy(),
#                                                                         time_period=self.train_val_test_periods['val'],
#                                                                         input_vars=list(self.ds.data_vars.keys()),
 
#                                                                         num_timesteps=self.num_timesteps,
#                                                                         region=self.region, 
#                                                                         scaling_mean_std=self.scaling_mean_std)
                    
#                 self.data_val = TWSA2DDataset(val_batches,
#                                                     target='twsa', 
#                                                 delay=self.horizon, 
#                                                 input_seq_len=self.local_lag,
#                                                 ds_oci=None, 
#                                                 oci_lag=None, 
#                                                 input_vars=None, 
#                                                 positional_vars=['longitude', 'latitude'], 
#                                                 oci_vars=None, 
#                                                 features_nanfill=-1.)
  
#         if stage == "test":
#                 # load datasets only if they're not loaded already
#             if not self.data_test:
#                 print("Scaling data for prediction with scaling mean std derived from training data")
#                 test_batches, test_oci_batches, scaling_mean_std = load_datasets(self.ds.copy(),
#                                                                         time_period=self.train_val_test_periods['test'],
#                                                                         input_vars=list(self.ds.data_vars.keys()),
#                                                                         # log_transform_vars =[None],
#                                                                         # oci_vars=[None],
#                                                                         num_timesteps=self.num_timesteps,
#                                                                         region=self.region, 
#                                                                         scaling_mean_std=self.scaling_mean_std)
                    
#                 self.data_test = TWSA2DDataset(test_batches,
#                                                     target='twsa', 
#                                                 delay=self.horizon, 
#                                                 input_seq_len=self.local_lag,
#                                                     ds_oci=None, 
#                                                     oci_lag=None, 
#                                                     input_vars=None, 
#                                                     positional_vars=['longitude', 'latitude'], 
#                                                     oci_vars=None, 
#                                                 features_nanfill=-1.)
#         if stage == "predict":
#                 # load datasets only if they're not loaded already
#                 print("Scaling data for prediction with scaling mean std derived from training data")
#                 predict_batches, pred_oci_batches, scaling_mean_std = load_datasets(self.ds.copy(),
#                                                                         time_period=self.train_val_test_periods['val'],
#                                                                         input_vars=list(self.ds.data_vars.keys()),
#                                                                         # log_transform_vars =[None],
#                                                                         # oci_vars=[None],
#                                                                         num_timesteps=self.num_timesteps,
#                                                                         region=self.region, 
#                                                                         scaling_mean_std=self.scaling_mean_std)
                    
#                 self.data_val = TWSA2DDataset(predict_batches,
#                                                     target='twsa', 
#                                                 delay=self.horizon, 
#                                                 input_seq_len=self.local_lag,
#                                                 time_step=0,
#                                                     ds_oci=None, 
#                                                     oci_lag=None, 
#                                                     input_vars=None, 
#                                                     positional_vars=['longitude', 'latitude'], 
#                                                     oci_vars=None,
#                                                     features_nanfill=-1.)

           




#     def train_dataloader(self):
#         #print(self.hparams.batch_size)
#         return DataLoader(
#             dataset=self.data_train,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=True,
#             persistent_workers=True,
#             drop_last=False
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             dataset=self.data_val,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=False,
#             persistent_workers=True,
#             drop_last=False
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             dataset=self.data_test,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=False,
#             persistent_workers=True,
#             drop_last=False
#         )
#     def predict_dataloader(self, time_step_idx: int, period=None):
#         if period is None:
#             period = self.train_val_test_periods['val']
#         predict_batches, pred_oci_batches, scaling_mean_std = load_datasets(self.ds.copy(),
#                                                                         time_period=period,
#                                                                         input_vars=list(self.ds.data_vars.keys()),
#                                                                         # log_transform_vars =[None],
#                                                                         # oci_vars=[None],
#                                                                         num_timesteps=self.num_timesteps,
#                                                                         region=self.region,
#                                                                         scaling_mean_std=self.scaling_mean_std)
                    
#         self.data_pred = TWSA2DDataset(predict_batches,
#                                                     target='twsa', 
#                                                     delay=self.horizon, 
#                                                     input_seq_len=self.local_lag,
#                                                     time_step=time_step_idx,
#                                                     ds_oci=None, 
#                                                     oci_lag=None, 
#                                                     input_vars=None, 
#                                                     positional_vars=['longitude', 'latitude'], 
#                                                     oci_vars=None,
#                                                     features_nanfill=-1.)
#         return DataLoader(dataset=self.data_pred,
#                     batch_size=self.hparams.batch_size,
#                     num_workers=self.hparams.num_workers,
#                     pin_memory=self.hparams.pin_memory,
#                     shuffle=False,
#                     persistent_workers=True,
#                     drop_last=False)
    
#     def __init__(
#             self,
#             ds_path: str = None,
#             input_vars: list = None,
#             local_lag: int = 12,
#             delay: int = 12,
#             target: str = 'twsa',
#             output_seq_len: int = 1, 
#             num_neighbours: int = 4,
#             log_transform_target: bool = True,
#             train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
#             **kwargs       
#             ):
#         super(TWSA_GraphDatamodule, self).__init__()

   
#         self.ds_path = ds_path
#         self.input_vars = input_vars
#         self.input_seq_len = local_lag
#         self.delay = delay
#         self.target = target
#         self.target_seq_len = output_seq_len
#         self.train_val_test_split = train_val_test_split
#         self.log_transform_target = log_transform_target
#         self.num_neighbours = num_neighbours

#         self.save_hyperparameters(logger=False)

#         self.ds = xr.open_dataset(ds_path)
#         if 'lat' not in self.ds.coords.keys():
#             self.ds = self.ds.rename({"latitude":"lat" , "longitude":"lon"}) # make sure naming conventions are consistent
#         if self.input_vars is None:
#             self.input_vars = list(self.ds.data_vars.keys())
#         # self.ds['sst'] = self.ds['sst'].where(self.ds['sst'] >= 0)

#         self.train_val_test_periods = self.get_train_val_test_periods()
#         self.static_features = self.get_static_feat() #needed for certain edge constructors
#         # land sea mask
#         # self.lsm = self.ds.lsm.values[0].copy() ## TODO maybe change this to .to_dataarray() since this only returns a pointer to the data right?
#         # self.lsm[29:] = np.zeros(self.lsm[29:].shape)      #mask out antarctica because it is not in the reconstr. TWSA data
#         # self.lsm = np.nan_to_num(self.lsm.reshape(-1), nan=0)
#         # self.lsm = np.where(self.lsm > 1e-3, 1, 0) # usually it should be binary but when coarsening or interpolating to a different resolution it may not be
    
#         self.edge_index = None
#         self.scaling_mean_std = None
#         self.ocean_clustering = None
#         self.land_clustering = None
#         self.data_train: Optional[Dataset] = None
#         self.data_val: Optional[Dataset] = None
#         self.data_test: Optional[Dataset] = None


#     def get_train_val_test_periods(self):     
#         self.total_time_steps = len(self.ds.time)
#         train_time_steps = int(self.total_time_steps * self.train_val_test_split[0])
#         val_time_steps = int(self.total_time_steps * self.train_val_test_split[1])

#         train_data_start = self.ds.time[0].values
#         train_data_end = self.ds.time[train_time_steps].values
#         val_data_start = self.ds.time[train_time_steps+1].values
#         val_data_end = self.ds.time[train_time_steps + val_time_steps].values
#         test_data_start = self.ds.time[train_time_steps + val_time_steps + 1].values
#         test_data_end = self.ds.time[-1].values
#         return {'train': (train_data_start, train_data_end),
#                 'val': (val_data_start, val_data_end),
#                 'test': (test_data_start, test_data_end)}

#     def get_static_feat(self, static_var=None):
#         if static_var is None:
#             # Get the values of the coordinates
#             latitude_values = torch.Tensor(self.ds.coords['lat'].values)
#             longitude_values = torch.Tensor(self.ds.coords['lon'].values)
#               # Create a 2D tensor from lat and lon values
#             lat_grid, lon_grid = torch.meshgrid(torch.tensor(latitude_values), torch.tensor(longitude_values))
#             static_feat = torch.stack([lon_grid, lat_grid ], dim=-1).reshape(-1, 2)  
#         else:  
#             static_feat = torch.stack([torch.Tensor(self.ds.isel(time=0)[var].values.reshape(-1)) for var in static_var], dim=-1)
#         return static_feat

          

#     def setup(self, stage: Optional[str] = None):
#         """ Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

#         This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
#         so be careful not to execute the random split twice! The `stage` can be used to
#         differentiate whether it's called before trainer.fit()` or `trainer.test()`.
#         """
        
#         if stage == "fit":
#             # load datasets only if they're not loaded already
#             if not self.data_train:                
#                 self.data_train = TWSAGraphDataset(self.ds.copy(), time_period=self.train_val_test_periods['train'],
#                                                     edge_index=self.edge_index,
#                                                     target='twsa', 
#                                                     delay_time=self.delay, 
#                                                     input_seq_len=self.input_seq_len,
#                                                     output_seq_len=self.target_seq_len,
#                                                     num_neighbours=self.num_neighbours,
#                                                     input_vars=list(self.ds.data_vars.keys()), 
#                                                     log_transform_target=self.log_transform_target,  
#                                                     features_nanfill=-1.)
                
#                 self.scaling_mean_std = self.data_train.scaling_params
#                 self.edge_index = self.data_train.edge_index
#                 self.ocean_clustering = self.data_train.ocean_clustering
#                 self.land_clustering = self.data_train.land_clustering
                

#             # Save mean std dict for normalization during inference time
#             #print("Mean", self.mean_std_dict)
#             #with open(f'mean_std_dict_{self.target_shift}.json', 'w') as f:
#             #    f.write(json.dumps(self.mean_std_dict))

        
#             # load datasets only if they're not loaded already
#             if not self.data_val:
#                 print("Scaling data for prediction with scaling mean std derived from training data")
#                 self.data_val = TWSAGraphDataset(self.ds.copy(), time_period=self.train_val_test_periods['val'],
#                                                     edge_index=self.edge_index,
#                                                     ocean_clustering=self.ocean_clustering,
#                                                     land_clustering=self.land_clustering,
#                                                     target='twsa', 
#                                                     delay_time=self.delay, 
#                                                     input_seq_len=self.input_seq_len,
#                                                     output_seq_len=self.target_seq_len,
#                                                     num_neighbours=self.num_neighbours,
#                                                     scaling_mean_std=self.scaling_mean_std,
#                                                     input_vars=list(self.ds.data_vars.keys()), 
#                                                     log_transform_target=self.log_transform_target,  
#                                                     features_nanfill=-1.)
#         if stage == "validate":
#                     # load datasets only if they're not loaded already
#             if not self.data_val:
#                 if self.scaling_mean_std is None:
#                     print("Mean and std not found, scaling data for validation with scaling mean std derived from validation data")
#                 self.data_val = TWSAGraphDataset(self.ds.copy(), time_period=self.train_val_test_periods['val'],
#                                                     edge_index=self.edge_index,
#                                                     ocean_clustering=self.ocean_clustering,
#                                                     land_clustering=self.land_clustering,
#                                                     target='twsa', 
#                                                     delay_time=self.delay, 
#                                                     input_seq_len=self.input_seq_len,
#                                                     output_seq_len=self.target_seq_len,
#                                                     num_neighbours=self.num_neighbours,
#                                                     scaling_mean_std=self.scaling_mean_std,
#                                                     input_vars=list(self.ds.data_vars.keys()), 
#                                                     log_transform_target=self.log_transform_target,  
#                                                     features_nanfill=-1.)
                
  
#         if stage == "test":
#                 # load datasets only if they're not loaded already
#             if not self.data_test:
#                 if self.scaling_mean_std is None:
#                     print("Mean and std not found, scaling data for test with scaling mean std derived from test data")
#                 self.data_test = TWSAGraphDataset(self.ds.copy(), time_period=self.train_val_test_periods['test'],
#                                                     edge_index=self.edge_index,
#                                                     ocean_clustering=self.ocean_clustering,
#                                                     land_clustering=self.land_clustering,
#                                                     target='twsa', 
#                                                     delay_time=self.delay, 
#                                                     input_seq_len=self.input_seq_len,
#                                                     output_seq_len=self.target_seq_len,
#                                                     num_neighbours=self.num_neighbours,
#                                                     scaling_mean_std=self.scaling_mean_std,
#                                                     input_vars=list(self.ds.data_vars.keys()), 
#                                                     log_transform_target=self.log_transform_target,  
#                                                     features_nanfill=-1.)
#         if stage == "predict":
#             pass

           


#     def unnormalize_twsa_1D(self, data):
#         assert self.scaling_mean_std is not None, "Mean and std not found"
#         assert self.scaling_mean_std[0][0].name == 'twsa', "Mean and std not found for twsa"
#         rescaled = data * torch.Tensor(self.scaling_mean_std[0][1].values).to(data.device) + torch.Tensor(self.scaling_mean_std[0][0].values).to(data.device)
#         if self.log_transform_target:
#             rescaled = torch.exp(rescaled)-1
#         return rescaled 
    
#     def unnormalize_twsa(self, data):
#         """Unnormalize the TWSA data using the mean and std from the training data.
#         data shapes: (batch_size*num_nodes, num_timesteps)
#         or data is an xr.DataArray
#         """
#         # num_nodes = self.get_num_nodes()
#         assert self.scaling_mean_std is not None, "Mean and std not found"
#         assert self.scaling_mean_std[0][0].name == 'twsa', "Mean and std not found for twsa"

#         if isinstance(data, torch.Tensor):
#             mean = torch.Tensor(self.scaling_mean_std[0][1].values).to(data.device)
#             std = torch.Tensor(self.scaling_mean_std[0][0].values).to(data.device)
#         else: 
#             mean = self.scaling_mean_std[0][1].values
#             std = self.scaling_mean_std[0][0].values

#         if mean.ndim == 1:  # simple normalization: mean/std over space and time         
#             rescaled = data * mean + std
#         elif mean.ndim == 2: # pixel-wise normalization: mean/std for each pixel over time  
#             lsm = self.get_land_ocean_mask()[0]     
#             if data.ndim == 2: # [batch_size*num_nodes, num_timesteps]
                
#                 batch_size = int(data.shape[0]/self.get_num_nodes())
#                 if isinstance(data, torch.Tensor):
#                     mean = torch.concat((mean[lsm], torch.zeros(self.num_ocean_clusters).to(data.device)), dim=0) # get into shape of nodes (land+ocean)
#                     std = torch.concat((std[lsm], torch.zeros(self.num_ocean_clusters).to(data.device)), dim=0)
            
#                     mean = mean.unsqueeze(1).repeat(batch_size, data.shape[-1]) # get into shape of timesteps
#                     std = std.unsqueeze(1).repeat(batch_size, data.shape[-1])
#                 else:
#                     mean = np.concatenate((mean[lsm], np.zeros(self.num_ocean_clusters)), axis=0)
#                     std = np.concatenate((std[lsm], np.zeros(self.num_ocean_clusters)), axis=0)
                    
#                     mean = np.tile(mean[:, None], (batch_size, data.shape[-1]))  
#                     std = np.tile(std[:, None], (batch_size, data.shape[-1]))  
#             elif data.ndim == 1:   
#                 if isinstance(data, torch.Tensor):
#                     mean = torch.concat((mean[lsm], torch.zeros(self.num_ocean_clusters).to(data.device)), dim=0) # get into shape of nodes (land+ocean)
#                     std = torch.concat((std[lsm], torch.zeros(self.num_ocean_clusters).to(data.device)), dim=0)
#                 else:
#                     mean = np.concatenate((mean[lsm], np.zeros(self.num_ocean_clusters)), axis=0)
#                     std = np.concatenate((std[lsm], np.zeros(self.num_ocean_clusters)), axis=0) 
#             elif data.ndim == 3:
#                 assert isinstance(data, xr.DataArray) , "3D data should be xr.DataArray for rescaling"


#             rescaled = data * mean + std
            
#         return rescaled 
    
#     def get_antarctica_landmask(self):
#         if self.data_train is not None:
#             return self.data_train.get_antarctica_landmask()
#         elif self.data_val is not None:
#             return self.data_val.get_antarctica_landmask()
#         elif self.data_test is not None:
#             return self.data_test.get_antarctica_landmask()
#         else:
#             return None

#     def train_dataloader(self):
#         #print(self.hparams.batch_size)
#         return DataLoader(
#             dataset=self.data_train,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             #pin_memory=self.hparams.pin_memory,
#             shuffle=True,
#             # persistent_workers=True,
#             drop_last=False,
#             collate_fn=collate_fn
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             dataset=self.data_val,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             #pin_memory=self.hparams.pin_memory,
#             shuffle=False,
#             # persistent_workers=True,
#             drop_last=False,
#             collate_fn=collate_fn
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             dataset=self.data_test,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             #pin_memory=self.hparams.pin_memory,
#             shuffle=False,
#             # persistent_workers=True,
#             drop_last=False,
#             collate_fn=collate_fn
#         )
#     def predict_dataloader(self, time_step_idx: int, period=None):
#         if period is None:
#             period = self.train_val_test_periods['val']
#             print("No time period specified. Predicting on validation period")
#         if self.scaling_mean_std is None:
#             print("Mean and std not found, scaling data for prediction with scaling mean std derived from prediction dataset")
#         self.data_pred = TWSAGraphDataset(self.ds.copy(), time_period=period, time_step=time_step_idx,
#                                                     edge_index=self.edge_index,
#                                                     ocean_clustering=self.ocean_clustering,
#                                                     land_clustering=self.land_clustering,
#                                                     target='twsa', 
#                                                     delay_time=self.delay, 
#                                                     input_seq_len=self.input_seq_len,
#                                                     output_seq_len=self.target_seq_len,
#                                                     num_neighbours=self.num_neighbours,
#                                                     scaling_mean_std=self.scaling_mean_std,
#                                                     input_vars=list(self.ds.data_vars.keys()), 
#                                                     log_transform_target=self.log_transform_target,  
#                                                     features_nanfill=-1.)
#         return DataLoader(dataset=self.data_pred,
#                     batch_size=self.hparams.batch_size,
#                     num_workers=self.hparams.num_workers,
#                     #pin_memory=self.hparams.pin_memory,
#                     shuffle=False,
#                     # persistent_workers=True,
#                     drop_last=False,
#             collate_fn=collate_fn)
    

# Function to load yaml configuration file

def read_config(config_path):
    import yaml
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config

if __name__ == '__main__':
    path= "/home/code/Seasfire_Causal/data/SeasFireCube_v3.zarr" #SeasFireCube_v3_1deg, SeasFireCube_v3
    ds=xr.open_zarr(path)
    print("data loaded")
    input_vars = ['t2m_mean','tp','vpd']
    positional_vars = ['cos_lon', 'cos_lat', 'sin_lon', 'sin_lat', 'latitude', 'longitude']
    oci_vars = ['oci_nao','oci_ao','oci_nina34_anom']
    data_module = TWSA_Datamodule(
            ds_path=path,
            input_vars=input_vars,
            log_transform_vars=[],
            positional_vars=positional_vars,
            oci_vars=oci_vars,
            local_lag= 39, #about 10 months (36~10months)
            oci_lag= 10, # at most 10 months, at each 31 day, may not align exactly to local input time steps, but within the similar periods, to make_oci_lag=10, local_lag has >= 39
            target='gwis_ba',
            target_shift=1,
            batch_size= 2,
            num_workers= 8,
            pin_memory = False,
            debug= True,
            region = {'longitude' : slice(-13, 45), 'latitude': slice(70, 25)},
            biomes = 3.0)
    data_module.setup()

    print(f'Training loader has a length of {len(data_module.data_train)}')
    it = data_module.data_train[0]
    for k,v in it.items():
        print(k,v.shape)
    print(it['times'])

    #x_local (3, 39)
    #x_local_pos (6,)
    #x_oci (3, 10)
    #y_local(1)
    train_loader = data_module.train_dataloader()
    train_loader = iter(train_loader)
    print(next(train_loader))
    import datetime
