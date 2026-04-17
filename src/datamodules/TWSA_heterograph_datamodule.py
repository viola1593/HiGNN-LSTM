from typing import Optional, Tuple, Dict
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import  DataLoader, Dataset
from torch_geometric.loader import DataLoader as pygDataLoader

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
try:
    from datamodules.TWSA_heterograph_dataset import TWSAGraphHeteroData, DummyDataset
except ModuleNotFoundError:
    from TWSA_heterograph_dataset import TWSAGraphHeteroData, DummyDataset

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def collate_fn(batch):
  
  return {
      'x_sequence': torch.stack([x['x_sequence'] for x in batch]),
      'y_timestep': torch.stack([x['y_timestep'] for x in batch]),
      'edge_index': torch.stack([x['edge_index'] for x in batch]),
}







class TWSA_HeteroGraphDatamodule(LightningDataModule):
    """
    Datamodule for the lightning dataset that returns a pyg.HeteroGraph data instance from __getitem__."""

    def __init__(
            self,
            ds_path: str = None,
            input_vars: list = None,
            input_seq_len: int = 12,
            delay: int = 12,
            target: str = 'twsa',
            land_threshold: float = 0.8, # threshold for land sea mask to define land pixels
            output_seq_len: int = 1, 
            k_mesh2mesh=10, 
            k_mesh2grid=1,
            log_transform_target: bool = False, # nonsense for negative values
            train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
            graph_path=None,
            num_ocean_clusters: int = 20,
            **kwargs       
            ):
        super(TWSA_HeteroGraphDatamodule, self).__init__()

   
        self.ds_path = ds_path
        self.input_seq_len = input_seq_len
        self.delay = delay
        self.target = target
        self.target_seq_len = output_seq_len
        self.k_mesh2mesh = k_mesh2mesh
        self.k_mesh2grid = k_mesh2grid
        self.train_val_test_split = train_val_test_split
        self.log_transform_target = log_transform_target
        self.graph_path=graph_path
        self.num_ocean_clusters = num_ocean_clusters

        self.save_hyperparameters(logger=False)

        self.ds = xr.open_dataset(ds_path)
        if 'lat' not in self.ds.coords.keys():
            self.ds = self.ds.rename({"latitude":"lat" , "longitude":"lon"}) # make sure naming conventions are consistent
        # land sea mask
        if 'lsm' in self.ds.data_vars.keys():
            self.land_ocean_mask = (self.ds['lsm']>land_threshold) # 1 is land, 0 is ocean, by changing resolution, land pixels could be <1 but never 0
        else:
            self.land_ocean_mask = None
            Warning("No land sea mask found in dataset.")
        if input_vars is not None:
            # subset the dataset to only include input variables
            self.ds = self.ds[input_vars] 
        self.input_vars = list(self.ds.data_vars.keys())
        
        self.num_features = len(self.input_vars)+2 # +2 for the temporal encoding
        self.only_land_variables = [ 'ro', 'pev', 'swvl1', 'swvl2', 'swvl3', 'swvl4','lai_hv', 'lai_lv']
        self.ocean_features = [item for item in self.input_vars if item not in self.only_land_variables+['twsa']]
        self.num_ocean_features = len(self.ocean_features)+2 # +2 for the temporal encoding
        
        self.train_val_test_periods = self.get_train_val_test_periods()
     
        self.edge_index = None
        self.scaling_mean_std = None
        self.ocean_clustering = None
        self.land_clustering = None
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
        test_data_start = self.ds.time[train_time_steps + val_time_steps + 1].values
        test_data_end = self.ds.time[-1].values
        
        return {'train': (train_data_start, train_data_end),
                'val': (val_data_start, val_data_end),
                # 'test': (val_data_start, val_data_end), # test data will be GRACE
                'test': (test_data_start, test_data_end)
                }

            

    def setup(self, stage: Optional[str] = None):
        """ Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        
        if stage == "fit":
            # load datasets only if they're not loaded already
            if not self.data_train:               
                self.data_train = TWSAGraphHeteroData(self.ds.copy(), time_period=self.train_val_test_periods['train'],
                                                    target='twsa', 
                                                    delay_time=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    k_mesh2mesh=self.k_mesh2mesh, 
                                                    k_mesh2grid=self.k_mesh2grid,
                                                    land_ocean_mask=self.land_ocean_mask,  
                                                    features_nanfill=0, 
                                                    graph_path=self.graph_path,
                                                    num_ocean_clusters=self.num_ocean_clusters,
                                                    train_val_test_data='train')
                
                self.scaling_mean_std = self.data_train.scaling_params
                self.ocean_clustering = self.data_train.ocean_clustering
                #self.land_clustering = self.data_train.land_clustering
                

            # Save mean std dict for normalization during inference time
            #print("Mean", self.mean_std_dict)
            #with open(f'mean_std_dict_{self.target_shift}.json', 'w') as f:
            #    f.write(json.dumps(self.mean_std_dict))

        
            # load datasets only if they're not loaded already
            if not self.data_val:
                print("Scaling data for prediction with scaling mean std derived from training data")
                self.data_val = TWSAGraphHeteroData(self.ds.copy(), time_period=self.train_val_test_periods['val'],
                                                    ocean_clustering=self.ocean_clustering,
                                                    #land_clustering=self.land_clustering,
                                                    target='twsa', 
                                                    delay_time=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    k_mesh2mesh=self.k_mesh2mesh, 
                                                    k_mesh2grid=self.k_mesh2grid,
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    land_ocean_mask=self.land_ocean_mask, 
                                                    features_nanfill=0,
                                                    graph_path=self.graph_path,
                                                    num_ocean_clusters=self.num_ocean_clusters, 
                                                    train_val_test_data='val')
        if stage == "validate":
                    # load datasets only if they're not loaded already
            if not self.data_val:
                if self.scaling_mean_std is None:
                    print("Mean and std not found, scaling data for validation with scaling mean std derived from validation data")
                self.data_val = TWSAGraphHeteroData(self.ds.copy(), time_period=self.train_val_test_periods['val'],
                                                    ocean_clustering=self.ocean_clustering,
                                                    #land_clustering=self.land_clustering,
                                                    target='twsa', 
                                                    delay_time=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    k_mesh2mesh=self.k_mesh2mesh, 
                                                    k_mesh2grid=self.k_mesh2grid,
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    land_ocean_mask=self.land_ocean_mask,  
                                                    features_nanfill=0,
                                                    graph_path=self.graph_path,
                                                    num_ocean_clusters=self.num_ocean_clusters,
                                                    train_val_test_data='val')
                
  
        if stage == "test":
                # load datasets only if they're not loaded already
            if not self.data_test:
                if self.scaling_mean_std is None:
                    print("Mean and std not found, scaling data for test with scaling mean std derived from test data")
                self.data_test = TWSAGraphHeteroData(self.ds.copy(), time_period=self.train_val_test_periods['test'],
                                                    ocean_clustering=self.ocean_clustering,
                                                    #land_clustering=self.land_clustering,
                                                    target='twsa', 
                                                    delay_time=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    k_mesh2mesh=self.k_mesh2mesh, 
                                                    k_mesh2grid=self.k_mesh2grid,
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    land_ocean_mask=self.land_ocean_mask,  
                                                    features_nanfill=0,
                                                    graph_path=self.graph_path,
                                                    num_ocean_clusters=self.num_ocean_clusters,
                                                    train_val_test_data='test')
        if stage == "predict":
            pass

           


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

        if std.ndim == 0: # simple normalization: mean/std over space and time         
            rescaled = data * std + mean
        elif std.ndim == 2: # pixel-wise normalization: mean/std for each pixel over time  

            if data.ndim == 2: # [batch_size*num_nodes, num_timesteps]
                batch_size = int(data.shape[0]/self.get_num_nodes())
                lsm = self.get_land_ocean_mask()[0].values     
                std = std[lsm]
                mean = mean[lsm]
                if isinstance(data, torch.Tensor):            
                    std = std.unsqueeze(1).repeat(batch_size, data.shape[-1]) # get into shape of timesteps
                    mean = mean.unsqueeze(1).repeat(batch_size, data.shape[-1])
                else:
                    std = np.tile(std[:, None], (batch_size, data.shape[-1]))  
                    mean = np.tile(mean[:, None], (batch_size, data.shape[-1]))  

            elif data.ndim == 3:
                assert isinstance(data, xr.DataArray) , "3D data should be xr.DataArray for rescaling"
            else: 
                lsm = self.get_land_ocean_mask()[0].values     
                std = std[lsm]
                mean = mean[lsm]

            rescaled = data * std + mean
            
        return rescaled 
    
    def get_num_nodes(self):
        num_nodes = self.data_train.num_land_nodes() if self.data_train is not None else self.data_val.num_land_nodes() if self.data_val is not None else self.data_test.num_land_nodes()
        return num_nodes

    def get_land_ocean_mask(self):
        if self.data_train is not None:
            return self.data_train.land_ocean_mask
        elif self.data_val is not None:
            return self.data_val.land_ocean_mask
        elif self.data_test is not None:
            return self.data_test.land_ocean_mask
        else:
            return None
    
    def get_antarctica_landmask(self):
        """True for pixels in antarctica"""
        if self.data_train is not None:
            return self.data_train.get_antarctica_landmask()
        elif self.data_val is not None:
            return self.data_val.get_antarctica_landmask()
        elif self.data_test is not None:
            return self.data_test.get_antarctica_landmask()
        else:
            return None
    def get_greenland_landmask(self):
        """True for pixels in greenland"""
        if self.data_train is not None:
            return self.data_train.get_greenland_landmask()
        elif self.data_val is not None:
            return self.data_val.get_greenland_landmask()
        elif self.data_test is not None:
            return self.data_test.get_greenland_landmask()
        else:
            return None

    def train_dataloader(self):
        #print(self.hparams.batch_size)
        return pygDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            #pin_memory=self.hparams.pin_memory,
            shuffle=True,
            # persistent_workers=True,
            drop_last=False
            
        )

    def val_dataloader(self):
        return pygDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            #pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # persistent_workers=True,
            drop_last=False,
            
        )

    def test_dataloader(self):
        return pygDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            #pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # persistent_workers=True,
            drop_last=False,
            
        )
    def predict_dataloader(self, time_step_idx: int=None, period=None):
        if period is None:
            period = self.train_val_test_periods['val']
            print("No time period specified. Predicting on validation period")
        if self.scaling_mean_std is None:
            print("Mean and std not found, scaling data for prediction with scaling mean std derived from prediction dataset")
        self.data_pred = TWSAGraphHeteroData(self.ds.copy(), time_period=period, time_step=time_step_idx,
                                                    ocean_clustering=self.ocean_clustering,
                                                    #land_clustering=self.land_clustering,
                                                    target='twsa', 
                                                    delay_time=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    k_mesh2mesh=self.k_mesh2mesh, 
                                                    k_mesh2grid=self.k_mesh2grid,
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    land_ocean_mask=self.land_ocean_mask,  
                                                    features_nanfill=0,
                                                    graph_path=self.graph_path,
                                                    num_ocean_clusters=self.num_ocean_clusters,
                                                    train_val_test_data='predict')
        return pygDataLoader(dataset=self.data_pred,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    #pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    # persistent_workers=True,
                    drop_last=False,
            )
    
    def GRACE_dataloader(self, stage='test', time_step_idx: int=None, train_split: float=0.9):
        """Dataloader that gets the dataset with GRACE-derived TWSA instead of the reconstructed TWSA"""
        grace_dataset = xr.open_dataset('data/latest_data/GRACE_era5_1deg_detrend_latavg_tempmatch.nc')
        
        if 'detrended_twsa' in grace_dataset.data_vars.keys():
            grace_dataset = grace_dataset.rename({"detrended_twsa":"twsa"}) # make sure naming conventions are consistent
        # get the dataset 
        
        # grace_dataset = grace_dataset.drop_vars(['time_bounds', 'lwe_thickness'])
        if grace_dataset['lat'][0] < grace_dataset['lat'][-1]:
            grace_dataset = grace_dataset.sortby('lat', ascending=False)
        
        lat = grace_dataset['lat'].values.copy()
        lat[0] = 89.5
        grace_dataset = grace_dataset.assign_coords(lat=lat)
        
        grace_dataset['twsa'] = xr.where(grace_dataset.lsm, grace_dataset['twsa'], 0.0) #mask out ocean pixels in TWSA data to avoid confusion for the model, since the input features also have ocean pixels masked out
        
        if self.input_vars is not None:
            # subset the dataset to only include input variables
            grace_dataset = grace_dataset[self.input_vars]
        
        
        if stage == 'test':
            # choose time slice that has monthly data without gaps (2018-11-16 to 2023-11-16) (GRACE-FO) 166:-1
            #30:101
            time_period = (self.data_train.time_period[-1], grace_dataset.time[-1].values) 
            # time_period = (grace_dataset.time[0].values, grace_dataset.time[-1].values) 
            self.data_pred = TWSAGraphHeteroData(grace_dataset, time_period=time_period, time_step=time_step_idx,
                                                    ocean_clustering=self.ocean_clustering,
                                                    target='twsa', 
                                                    delay_time=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    k_mesh2mesh=self.k_mesh2mesh, 
                                                    k_mesh2grid=self.k_mesh2grid,
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    land_ocean_mask=self.land_ocean_mask, 
                                                    features_nanfill=0,
                                                    graph_path=self.graph_path,
                                                    num_ocean_clusters=self.num_ocean_clusters,
                                                    train_val_test_data='grace')
            return pygDataLoader(dataset=self.data_pred,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False,
                    drop_last=False)
        
        elif stage == 'fit':
            time_period = (grace_dataset.time.values[0], self.data_train.time_period[-1]) 
            # check how many samples we can get
            sequence_counter = 0
            index_list =[]
            for i, start_time in enumerate(grace_dataset.time.values):
                end_time = start_time.astype('datetime64[M]') + np.timedelta64(self.input_seq_len + self.delay + self.target_seq_len+ self.delay-1, 'M')
                if end_time > time_period[1]:
                    break
                else: 
                    sequence = grace_dataset.sel(time=slice(start_time, end_time))
                    if sequence.time.shape[0] < self.input_seq_len + self.delay + self.target_seq_len:
                        continue
                    elif sequence.time[-1]  == sequence.time[0].values.astype('datetime64[M]') + np.timedelta64(self.input_seq_len + self.delay + self.target_seq_len-1, 'M'):
                        # print(f"Found sequence from {start_time} to {end_time} with {sequence.time.shape[0]} time steps")
                        sequence_counter += 1
                        index_list.append(i)
            last_train_index=index_list[int(len(index_list)*train_split)-1]


            train_period = (time_period[0], grace_dataset.time[last_train_index].values)  # mostly GRACE

            
            self.data_train = TWSAGraphHeteroData(grace_dataset, time_period=train_period, time_step=time_step_idx,
                                                    ocean_clustering=self.ocean_clustering,
                                                    target='twsa', 
                                                    delay_time=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    k_mesh2mesh=self.k_mesh2mesh, 
                                                    k_mesh2grid=self.k_mesh2grid,
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    land_ocean_mask=self.land_ocean_mask, 
                                                    features_nanfill=0,
                                                    graph_path=self.graph_path,
                                                    num_ocean_clusters=self.num_ocean_clusters,
                                                    train_val_test_data='grace_train')
            # self.scaling_mean_std = self.data_train.scaling_params
            
            val_period = (grace_dataset.time[last_train_index].values, time_period[1])  
            self.data_val = TWSAGraphHeteroData(grace_dataset, time_period=val_period, time_step=time_step_idx,
                                                    ocean_clustering=self.ocean_clustering,
                                                    target='twsa', 
                                                    delay_time=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    k_mesh2mesh=self.k_mesh2mesh, 
                                                    k_mesh2grid=self.k_mesh2grid,
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    land_ocean_mask=self.land_ocean_mask, 
                                                    features_nanfill=0,
                                                    graph_path=self.graph_path,
                                                    num_ocean_clusters=self.num_ocean_clusters,
                                                    train_val_test_data='grace')
            
            train_dl = pygDataLoader(dataset=self.data_train,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=True,
                    drop_last=False)
            val_dl = pygDataLoader(dataset=self.data_val,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False,
                    drop_last=False)
            
            return train_dl, val_dl 
    
    
    
    def dummy_dataloader(self):
        self.data_dummy = DummyDataset(self.ds.copy(), time_period=self.train_val_test_periods['test'], time_step=0,
                                                    ocean_clustering=self.ocean_clustering,
                                                    target='twsa', 
                                                    delay_time=self.delay, 
                                                    input_seq_len=self.input_seq_len,
                                                    output_seq_len=self.target_seq_len,
                                                    k_mesh2mesh=self.k_mesh2mesh, 
                                                    k_mesh2grid=self.k_mesh2grid,
                                                    scaling_mean_std=self.scaling_mean_std,
                                                    land_ocean_mask=self.land_ocean_mask, 
                                                    features_nanfill=0,
                                                    graph_path=self.graph_path,
                                                    num_ocean_clusters=self.num_ocean_clusters,
                                                    train_val_test_data='dummy')
        return pygDataLoader(dataset=self.data_dummy,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    #pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    # persistent_workers=True,
                    drop_last=False,
            )

def float2datetime(float_values):
    return (float_values *np.timedelta64(1, 'D') + np.datetime64('1970-01-01T00:00:00'))
# Function to load yaml configuration file

def read_config(config_path):
    import yaml
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config
if __name__ == '__main__':
    config_path = 'configs/config_heterograph.yaml'
    config = read_config(config_path)
    datamod = TWSA_HeteroGraphDatamodule(**config['datamodule'])
    
    datamod.setup(stage='fit')
    train_dataloader = datamod.train_dataloader()
    for b in train_dataloader:
        print(b[0]['grid'].x)
    import pdb; pdb.set_trace()
    grace_dl = datamod.GRACE_dataloader(stage='test')
    for batch in grace_dl:
        print(batch[0]['grid'].x)

    for i in range(11):
        plt.plot(batch[0]['grid'].x[:,:,i])
        plt.savefig(f'newgrace_x{i}.png')
        plt.close()
        plt.plot(b[0]['grid'].x[:,:,i])
        plt.savefig(f'newtrain_x{i}.png')
        plt.close()
    import pdb; pdb.set_trace()
    feat, label = next(iter(train_dataloader))
    datamod.setup(stage='validate')
    datamod.setup(stage='test')