from typing import Optional, Tuple, Dict
import json
import torch
import warnings
from pytorch_lightning import LightningDataModule
from torch.utils.data import  Dataset
from torch_geometric.loader import DataLoader as pygDataLoader

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from pathlib import Path
try:
    from datamodules.TWSA_heterograph_dataset import TWSAGraphHeteroData
except ModuleNotFoundError:
    from TWSA_heterograph_dataset import TWSAGraphHeteroData

GRAPH_VARIANTS = ['full', 'no_ocean', 'local_ocean']








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
            graph_variant: str = 'full',
            graph_embedding_version: str = 'unitsphere_v2',
            num_ocean_clusters: int = 20,
            twsa_anomaly: bool = False, # models see and predict twsa - C_train[month]
            era5_anomaly: bool = False, # models see every input except twsa as var - C_train[month]
            **kwargs
            ):
        super(TWSA_HeteroGraphDatamodule, self).__init__()

        assert 'graph_path' not in kwargs, ("graph_path is derived from graph_variant and "
                                            "graph_embedding_version, remove it from the config")
        assert graph_variant in GRAPH_VARIANTS, f"graph_variant must be one of {GRAPH_VARIANTS}, got {graph_variant}"

        self.ds_path = ds_path
        self.input_seq_len = input_seq_len
        self.delay = delay
        self.target = target
        self.target_seq_len = output_seq_len
        self.k_mesh2mesh = k_mesh2mesh
        self.k_mesh2grid = k_mesh2grid
        self.train_val_test_split = train_val_test_split
        self.log_transform_target = log_transform_target
        self.graph_variant = graph_variant
        self.graph_embedding_version = graph_embedding_version
        self.graph_path = f"graphs/{graph_variant}_{graph_embedding_version}"
        self.use_ocean_nodes = graph_variant != 'no_ocean' # the model has to be built to match the graph
        self.local_ocean_coupling = graph_variant == 'local_ocean' # ocean->basin edges only to the nearest clusters
        self.num_ocean_clusters = num_ocean_clusters
        self.twsa_anomaly = twsa_anomaly
        self.era5_anomaly = era5_anomaly
        self._ensure_graph_meta()

        self.save_hyperparameters(logger=False)

        self.ds = xr.open_dataset(ds_path)
        if 'lat' not in self.ds.coords.keys():
            self.ds = self.ds.rename({"latitude":"lat" , "longitude":"lon"}) # make sure naming conventions are consistent
        # land sea mask
        if 'lsm' in self.ds.data_vars.keys():
            self.land_ocean_mask = (self.ds['lsm']>land_threshold) # 1 is land, 0 is ocean, by changing resolution, land pixels could be <1 but never 0
        else:
            self.land_ocean_mask = None
            warnings.warn("No land sea mask found in dataset.")
        if input_vars is not None:
            # subset the dataset to only include input variables
            self.ds = self.ds[input_vars] 
        self.input_vars = list(self.ds.data_vars.keys())
        
        self.num_features = len(self.input_vars)+2 # +2 for the temporal encoding
        self.only_land_variables = [ 'ro', 'pev', 'swvl1', 'swvl2', 'swvl3', 'swvl4','lai_hv', 'lai_lv']
        self.ocean_features = [item for item in self.input_vars if item not in self.only_land_variables+['twsa']]
        self.num_ocean_features = len(self.ocean_features)+2 # +2 for the temporal encoding
        
        self.train_val_test_periods = self.get_train_val_test_periods()
        # raw training climatology [month, lat, lon] in cm, also needed by eval in both modes.
        # float32 so it does not promote the datasets' twsa, NaN (Antarctica, ocean) -> 0
        self.twsa_climatology = (self.ds['twsa'].sel(time=slice(*self.train_val_test_periods['train']))
                                 .groupby('time.month').mean('time').fillna(0).astype(np.float32))
        # same for every other input (the ERA5 variables), NaN outside a variable's domain -> 0.
        # only built when used: with input_vars=['twsa'] there is nothing to average
        self.era5_climatology = (self.ds[[v for v in self.input_vars if v != 'twsa']]
                                 .sel(time=slice(*self.train_val_test_periods['train']))
                                 .groupby('time.month').mean('time').fillna(0).astype(np.float32)
                                 if era5_anomaly else None)

        self.scaling_mean_std = None
        self.ocean_clustering = None
        self.land_clustering = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def _ensure_graph_meta(self):
        '''Make sure the cached graph files in self.graph_path were built for this variant,
        embedding version, dataset and number of ocean clusters. The cached edge indices are
        keyed only by k but their node indices depend on the mesh size
        (n_basins + num_ocean_clusters), so without this check a stale cache would be loaded
        silently.'''
        meta = {'variant': self.graph_variant,
                'embedding_version': self.graph_embedding_version,
                'ds_path': self.ds_path,
                'num_ocean_clusters': self.num_ocean_clusters}
        meta_file = Path(self.graph_path) / 'meta.json'
        if not Path(self.graph_path).exists():
            Path(self.graph_path).mkdir(parents=True)
            meta_file.write_text(json.dumps(meta, indent=2))
            print(f"Created new graph directory {self.graph_path} with {meta}")
            return
        assert meta_file.exists(), (f"Graph directory {self.graph_path} exists but has no meta.json - "
                                    "it was built before graph variants were tracked. Delete it or "
                                    "choose a different graph_variant/graph_embedding_version.")
        cached_meta = json.loads(meta_file.read_text())
        assert cached_meta == meta, (f"Cached graph in {self.graph_path} was built with {cached_meta}, "
                                     f"but this run expects {meta}")
        print(f"Using cached graph directory {self.graph_path} with {cached_meta}")


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
                                                    use_ocean_nodes=self.use_ocean_nodes,
                                                    local_ocean_coupling=self.local_ocean_coupling,
                                                    twsa_climatology=self.twsa_climatology if self.twsa_anomaly else None,
                                                    era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                                    train_val_test_data='train')
                
                self.scaling_mean_std = self.data_train.scaling_params
                self.ocean_clustering = self.data_train.ocean_clustering

                


        
            # load datasets only if they're not loaded already
            if not self.data_val:
                print("Scaling data for prediction with scaling mean std derived from training data")
                self.data_val = TWSAGraphHeteroData(self.ds.copy(), time_period=self.train_val_test_periods['val'],
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
                                                    use_ocean_nodes=self.use_ocean_nodes,
                                                    local_ocean_coupling=self.local_ocean_coupling,
                                                    twsa_climatology=self.twsa_climatology if self.twsa_anomaly else None,
                                                    era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                                    train_val_test_data='val')
        if stage == "validate":
                    # load datasets only if they're not loaded already
            if not self.data_val:
                if self.scaling_mean_std is None:
                    print("Mean and std not found, scaling data for validation with scaling mean std derived from validation data")
                self.data_val = TWSAGraphHeteroData(self.ds.copy(), time_period=self.train_val_test_periods['val'],
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
                                                    use_ocean_nodes=self.use_ocean_nodes,
                                                    local_ocean_coupling=self.local_ocean_coupling,
                                                    twsa_climatology=self.twsa_climatology if self.twsa_anomaly else None,
                                                    era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                                    train_val_test_data='val')
                
  
        if stage == "test":
                # load datasets only if they're not loaded already
            if not self.data_test:
                if self.scaling_mean_std is None:
                    print("Mean and std not found, scaling data for test with scaling mean std derived from test data")
                self.data_test = TWSAGraphHeteroData(self.ds.copy(), time_period=self.train_val_test_periods['test'],
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
                                                    use_ocean_nodes=self.use_ocean_nodes,
                                                    local_ocean_coupling=self.local_ocean_coupling,
                                                    twsa_climatology=self.twsa_climatology if self.twsa_anomaly else None,
                                                    era5_climatology=self.era5_climatology if self.era5_anomaly else None,
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
        return pygDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=False
            
        )

    def val_dataloader(self):
        return pygDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            
        )

    def test_dataloader(self):
        return pygDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
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
                                                    use_ocean_nodes=self.use_ocean_nodes,
                                                    local_ocean_coupling=self.local_ocean_coupling,
                                                    twsa_climatology=self.twsa_climatology if self.twsa_anomaly else None,
                                                    era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                                    train_val_test_data='predict')
        return pygDataLoader(dataset=self.data_pred,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False,
                    drop_last=False,
            )
    
    def GRACE_dataloader(self, stage='test', time_step_idx: int=None, twsa_climatology=None):
        """Dataloader that gets the dataset with GRACE-derived TWSA instead of the reconstructed TWSA.
        twsa_climatology: [month, lat, lon] used instead of the training climatology for the twsa
            input anomaly and twsa_offset (anomaly mode only). ERA5 climatology and scaling stay the training ones."""
        if twsa_climatology is not None and not self.twsa_anomaly:
            raise ValueError("twsa_climatology needs twsa_anomaly=True")
        if twsa_climatology is None:
            twsa_climatology = self.twsa_climatology
        grace_dataset = xr.open_dataset('data/processed/GRACE_era5.nc')
        
        if 'detrended_twsa' in grace_dataset.data_vars.keys():
            grace_dataset = grace_dataset.rename({"detrended_twsa":"twsa"}) # make sure naming conventions are consistent
    
        if grace_dataset['lat'][0] < grace_dataset['lat'][-1]:
            grace_dataset = grace_dataset.sortby('lat', ascending=False)
        
        grace_dataset['twsa'] = xr.where(grace_dataset.lsm, grace_dataset['twsa'], 0.0) #mask out ocean pixels in TWSA data to avoid confusion for the model, since the input features also have ocean pixels masked out
        
        if self.input_vars is not None:
            # subset the dataset to only include input variables
            grace_dataset = grace_dataset[self.input_vars]
        
        
        if stage == 'test':
            # choose time slice that has no overlap with training dataset (2008-03 to 2023-11-16)  
            time_period = (self.data_train.time_period[-1], grace_dataset.time[-1].values) 
            # data gaps will be handled by the dataset class
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
                                                    use_ocean_nodes=self.use_ocean_nodes,
                                                    local_ocean_coupling=self.local_ocean_coupling,
                                                    twsa_climatology=twsa_climatology if self.twsa_anomaly else None,
                                                    era5_climatology=self.era5_climatology if self.era5_anomaly else None,
                                                    train_val_test_data='grace')
            return pygDataLoader(dataset=self.data_pred,
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False,
                    drop_last=False)
        



