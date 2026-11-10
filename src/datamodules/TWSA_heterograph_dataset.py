import copy
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import shapely
import sys
import torch
import torch_geometric as pyg
import warnings
import xarray as xr

from datetime import datetime
from pathlib import Path
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.utils import add_remaining_self_loops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import plot_graph_3d


def log_cache_hit(path):
    '''Report which cached graph file was loaded and when it was written, so that a stale cache
    is visible in the logs instead of being used silently.'''
    mtime = datetime.fromtimestamp(Path(path).stat().st_mtime).strftime('%m-%d-%Y_%H:%M:%S')
    print(f"Loaded cached graph file {path} (last modified {mtime})")






# create cell dataset
class TWSAGraphDataset(Dataset):
    def __init__(self, ds, time_period,  
                 ocean_clustering=None,
                 num_ocean_clusters=20,
                 land_clustering=None,
                 target='twsa', delay_time=12, input_seq_len=12, output_seq_len=1, 
                 scaling_mean_std=None,
                 num_neighbours=10,
                 time_step=None,
                 positional_vars=None, 
                 land_ocean_mask: xr.DataArray=None,
                 features_nanfill=0,
                 graph_path=None,
                 train_val_test_data='train',
                 use_ocean_nodes=True,
                 local_ocean_coupling=False,
                 twsa_climatology: xr.DataArray=None,
                 era5_climatology: xr.Dataset=None):
        """
        ds: xarray dataset with coords (latitude, longitude, time)
        time_period: tuple with start and end time of the dataset
        edge_index: edge index of the graph, if None, it will be created
        input_seq_len: number of time steps in the feature sequence 
        output_seq_len: number of time steps in the target sequence
        delay_time: number of time steps between end of sequence and target
        time_step: provide a specific time step to only get the data for that time step for prediction (default None)
        input_vars: variables to include in the input sequence
        positional_vars: name of the positional coordinates in the dataset
        features_nanfill: value to fill nan values in the features
        use_ocean_nodes: if False, no ocean clustering is done and the dataset has no ocean nodes
            (no_ocean graph variant)
        local_ocean_coupling: if True, ocean cluster nodes only send messages to their nearest
            basins instead of to all of them (local_ocean graph variant)
        twsa_climatology: training climatology [month, lat, lon] in cm. If given, twsa is normalized
            as the anomaly twsa - C[month] and each sample carries C of its target months as twsa_offset
        era5_climatology: training climatology with one [month, lat, lon] variable per input except twsa.
            If given, each of its variables is normalized as the anomaly var - C[var][month]
        """
        super(TWSAGraphDataset, self).__init__()

        self.train_val_test_data = train_val_test_data
        self.time_period = time_period
        self.ds_timeslice = ds.sel(time=slice(time_period[0], time_period[1]))


         # Extract basin IDs if present in dataset
        if 'basin_id' in ds:
            self.basin_ids = self.ds_timeslice['basin_id']  # Store basin IDs
            self.ds_timeslice = self.ds_timeslice.drop_vars('basin_id')  # Remove from dataset
        else:
            self.basin_ids = None
        
        self.ocean_clustering = ocean_clustering
        self.num_ocean_clusters = num_ocean_clusters
        self.use_ocean_nodes = use_ocean_nodes
        self.local_ocean_coupling = local_ocean_coupling
        self.land_clustering = land_clustering
        self.target = target
        self.delay = delay_time
        
        self.seq_len = input_seq_len
        self.pred_len = output_seq_len

        self.scaling_params = scaling_mean_std
        self.num_neighbours = num_neighbours
        

        self.positional_vars = positional_vars
        
        self.features_nanfill = features_nanfill
        self.graph_path = graph_path
        self.time_step = time_step

        
        # convert the values to float32 to save memory
        self.ds_timeslice = self.ds_timeslice.astype(np.float32)


        # get the land ocean mask before normalizing the data
        self.land_ocean_mask = land_ocean_mask # 1 is land, 0 is ocean, by changing resolution, land pixels could be <1 but never 0
        self.input_vars = list(self.ds_timeslice.data_vars.keys())
        self.target_index = self.input_vars.index(self.target)


        self.only_land_variables = [ 'ro', 'pev', 'swvl1', 'swvl2', 'swvl3', 'swvl4','lai_hv', 'lai_lv']
        self.only_ocean_variables = ['sst']

        self.twsa_offset = None
        if twsa_climatology is not None:
            # anomaly space: the per-pixel normalization below then scales twsa - C[month]
            clim = twsa_climatology.sel(month=self.ds_timeslice.time.dt.month).drop_vars('month')
            self.ds_timeslice['twsa'] = self.ds_timeslice['twsa'] - clim
            # C on the land nodes in separate_land_ocean order, (num_nodes, num_time_steps) like target_series
            self.twsa_offset = torch.from_numpy(clim.values[:, self.land_ocean_mask.isel(time=0).values]).T

        if era5_climatology is not None:
            for var in era5_climatology.data_vars:
                self.ds_timeslice[var] = (self.ds_timeslice[var]
                                          - era5_climatology[var].sel(month=self.ds_timeslice.time.dt.month).drop_vars('month'))

        self.normalize_ds_timeslice()
        self.month_to_features()

        self.land, self.ocean, _ = self.separate_land_ocean()
        if self.use_ocean_nodes:
            self.coarse_ocean = self.cluster_pixels(self.ocean, var_list=self.input_vars,
                                                          num_timesteps=len(self.ds_timeslice.time),
                                                          pixel_type='ocean')
        else:
            # no_ocean variant: no ocean cluster nodes at all, empty arrays keep the shapes below valid
            self.coarse_ocean = (np.zeros((0, 2)),
                                 np.zeros((len(self.input_vars), len(self.ds_timeslice.time), 0)))

                   
        self.samples = self.get_sample_indices()
        # get the coordinates of the nodes after coarsening
        self.coords = np.concatenate([self.land[0], self.coarse_ocean[0]], axis=0)
        self.features = torch.Tensor(self.land[1]).permute(2, 1, 0) # (num_nodes, num_time_steps, num_variables)
        self.target_series =  self.features[:, : , self.target_index] # (num_nodes, num_time_steps)
        self.ocean_features = [item for item in self.input_vars if item not in self.only_land_variables+['twsa']]
        self.ocean_indices = [i for i, item in enumerate(self.input_vars) if item in self.ocean_features]
        self.ocean_features = torch.Tensor(self.coarse_ocean[1]).permute(2, 1, 0)[:,:,self.ocean_indices]


    def get_sample_indices(self):
        if self.time_step is not None:
            # this is for inference when we only want to predict one time step
            t = self.time_step
            if t < self.seq_len:
                print(f"Time step {t} is before the local lag {self.seq_len}, setting time step to {t+self.seq_len}")
                t += self.seq_len
            if t > len(self.ds_timeslice.time) - self.delay - self.pred_len:
                print(f"Time step {t} is after the last possible time step, setting time step to {len(self.ds_timeslice.time) - self.delay -self.pred_len}")
                t = len(self.ds_timeslice.time) - self.delay - self.pred_len
            feature_indices_list = [t]
      
        else: 
            # range(sequence length (time steps we need before the first prediction) to len(time) - delay (time steps we need after the last prediction))
            feature_indices_list = [t for t in range(self.seq_len, len(self.ds_timeslice.time)-self.delay-self.pred_len+1)]
            
            feature_indices_list = self.filter_samples(feature_indices_list)
        return feature_indices_list

    def filter_samples(self, indices):
        """ filter the sample indices to only includes indices where the time series has regular timesteps (e.g. no missing months, no double months)"""
        
        time_values = self.ds_timeslice.time.values
        filtered_indices = []
        if self.delay==0: 
            for t in indices:
                time_steps = time_values[t - self.seq_len : t + self.pred_len + self.delay]
                # check if the time steps are regular with a monthly frequency, we allow for some numerical imprecision with np.isclose and a tolerance of 1 day
                end_time_step = time_steps[0].astype('datetime64[M]') + np.timedelta64(self.seq_len+self.pred_len + self.delay -1, 'M') # the last time step should be the first time step + the number of time steps in the input and output sequence (minus one because we count the first time step as well)
                if np.array_equal(time_steps[-1].astype('datetime64[M]'), end_time_step):
                    filtered_indices.append(t)
        else:
            # TODO: implement filtering for delay > 0, we do not need regular time steps between the end of the input sequence and the start of the target sequence, 
            # but we do need regular time steps within the input sequence and within the target sequence
            warnings.warn("Filtering of samples with irregular time steps is currently only implemented for delay=0, skipping filtering")
        return filtered_indices

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
    
    def normalize_ds_timeslice(self): 
        # normalize input variables 
        temp_dim = self.ds_timeslice.time.size
        mask_land = self.land_ocean_mask.isel(time=slice(0, temp_dim)).values # 1 is land and 0 is ocean
        mask_ocean = ~mask_land
        
        landmask_no_antarctica = (self.land_ocean_mask.where(self.ds_timeslice['lat']>-60)).isel(time=slice(0, temp_dim)).fillna(0).astype('bool').values
           
        if self.scaling_params is not None:
            # NaNs are filled after normalization and masking (see below), the same order as in
            # the training branch, so that a NaN inside the valid mask is encoded identically
            for i, var in enumerate(self.input_vars):
                try:
                    assert var == self.scaling_params[i][0].name
                    assert var == self.scaling_params[i][1].name
                    self.ds_timeslice[var] = (self.ds_timeslice[var] - self.scaling_params[i][0]) / self.scaling_params[i][1]
                except (AssertionError, AttributeError): print(f"Variable {var} not found in scaling parameters, skipping normalization")
                
                if var in self.only_land_variables:
                    mask = mask_land
                elif var in self.only_ocean_variables:
                    mask = mask_ocean
                elif var== 'twsa':
                    mask = landmask_no_antarctica
                    
                else:
                    mask = np.ones_like(mask_land).astype('bool')
                self.ds_timeslice[var].values[~mask] = 0 #self.ds_timeslice[var].where(~mask, 0).fillna(self.features_nanfill)
                self.ds_timeslice[var] = self.ds_timeslice[var].fillna(self.features_nanfill)
        
        else:
            scaling_mean_std =[]
            for var in self.input_vars:
                if var in self.only_land_variables:
                    mask = mask_land
                elif var in self.only_ocean_variables:
                    mask = mask_ocean
                elif var== 'twsa':
                    mask = landmask_no_antarctica
                elif var == 'lsm':
                    scaling_mean_std.append((0, 1))
                    continue # do not normalize land sea mask
                else:
                    mask = np.ones_like(mask_land).astype('bool')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # Compute mean and std for normalization over time per pixel
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

    
    def unnormalize(self, data):
        raise NotImplementedError("Unnormalization not implemented yet")
    
    def separate_land_ocean(self):
        '''separates the dataset into land and ocean numpy arrays
        land_sea_mask: name of the land sea mask variable in the dataset
        returns: 
            (land_coords, land_features): shape ((num_lat_lon,), (num_features, num_time_steps, num land grid cells)),
            (ocean_coords, ocean_features): shape ((num_lat_lon,), (num_features, num_time_steps, num ocean grid cells)),
            list of variable names from the original dataset for indexing the features in further processing'''
        num_features = len(self.ds_timeslice.data_vars)
        num_timesteps = len(self.ds_timeslice.time)

        land_sea_mask = self.land_ocean_mask.isel(time=slice(0, num_timesteps)) # due to coarsening in the preprocessing step, the land_sea_mask is not binary
        print('Shape of the land_sea_mask', land_sea_mask.shape) # (num_time_steps, num_lat, num_lon)
       
        # Get a latitude and longitude grid
        grid_lat = np.array(
                self.ds_timeslice["lat"], dtype=np.float32
            )  # (num_lat,)
        grid_lon = np.array(
                self.ds_timeslice["lon"], dtype=np.float32
            )  # (num_lon,)
        grid_lon_lat = np.stack(
                (np.expand_dims(grid_lon, 0).repeat(
                        len(grid_lat), 0
                    ),  # (num_lon, num_lat)
                    np.expand_dims(grid_lat, 1).repeat(
                        len(grid_lon), 1
                    ),  # (num_lat, num_long)
                ),axis=2, )  # (num_lon*num_lat, 2)

        # materialize the (num_features, num_time_steps, num_lat, num_lon) array once and index
        # it for both masks, instead of building it separately for land and for ocean
        all_features = self.ds_timeslice.to_dataarray().values

        # Get the land grid cells
        land_coords = grid_lon_lat[land_sea_mask[0]] # index 0 to get the first time step, lat_lon grid has no time component
        land_features = all_features[:,land_sea_mask] # (num_features, num_time_steps*num_lat_lon)
        land_features = land_features.reshape( num_features, num_timesteps, -1) # (num_features, num_time_steps,  number of land grid cells)
        # Do the same for the ocean grid cells
        ocean_coords = grid_lon_lat[~land_sea_mask[0]] # index 0 to get the first time step, lat_lon grid has no time component
        ocean_features = all_features[:,~land_sea_mask] # (num_features, num_time_steps*num_lat_lon)
        ocean_features = ocean_features.reshape(num_features, num_timesteps,-1) # (num_features, num_time_steps,  number of ocean grid cells)

        # check if the land and ocean nodes sum up to the total number of grid cells
        assert land_coords.shape[0] + ocean_coords.shape[0] == self.ds_timeslice.sizes['lat']*self.ds_timeslice.sizes['lon']
        assert land_features.shape[2] + ocean_features.shape[2] == self.ds_timeslice.sizes['lat']*self.ds_timeslice.sizes['lon']
        
        return (land_coords, land_features), (ocean_coords, ocean_features), list(self.ds_timeslice.data_vars)
    
    def cluster_pixels(self, pixels, var_list, num_timesteps,
                       pixel_type='ocean'):


        assert self.graph_path is not None, "a graph_path is required to cache the clustering model and save the cluster plot"

        if self.ocean_clustering is not None:
            print('Using predefined clustering model')
            assert len(self.ocean_clustering) == len(pixels[0]), \
                f"predefined clustering has {len(self.ocean_clustering)} labels, expected one per ocean pixel ({len(pixels[0])})"
        else:
            # prepare the inputs for the clustering, only needed to fit or apply a clustering model
            k_means_input = self.prepare_inputs_clustering(pixels)
            try:
                with open(self.graph_path+f"/{pixel_type}_clustering_model_{self.num_ocean_clusters}.pkl", "rb") as f:
                    pipe = pickle.load(f)
                log_cache_hit(self.graph_path+f"/{pixel_type}_clustering_model_{self.num_ocean_clusters}.pkl")

            except FileNotFoundError:
                print('Error could not load clustering model, creating a new one')
        
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('kmeans', KMeans(n_clusters=self.num_ocean_clusters, random_state=42))
                ])
                pipe.fit(k_means_input)
                print('num iterations:', pipe['kmeans'].n_iter_)
                print('number of clusters:', len(np.unique(pipe['kmeans'].labels_)))
                # save the model to a file
                Path(self.graph_path).mkdir(parents=True, exist_ok=True)
                with open(self.graph_path+f"/{pixel_type}_clustering_model_{self.num_ocean_clusters}.pkl", "wb") as f:
                    pickle.dump(pipe, f)
                            # Get the cluster labels for each ocean grid cell
            self.ocean_clustering = pipe.predict(k_means_input)
            
   

        # Initialize arrays to store the coarsened ocean coordinates and features
        coarsened_coords = np.zeros((self.num_ocean_clusters, 2))
        coarsened_features = np.zeros((len(var_list), num_timesteps, self.num_ocean_clusters))
        # Aggregate the coordinates and features for each cluster
        ocean_coords, ocean_features = pixels

        #plot ocean clusters
        plt.scatter(ocean_coords[:,0], ocean_coords[:,1], c=self.ocean_clustering, cmap='tab20', s=1)
        plt.title(f'Ocean clusters: {self.num_ocean_clusters}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(label='Cluster Label')
        plt.savefig(self.graph_path+f'/{pixel_type}_clusters_{self.train_val_test_data}.png')
        plt.close()
        
        for cluster in range(self.num_ocean_clusters):
            cluster_mask = (self.ocean_clustering == cluster)
            # the clusters are fitted on the unit sphere and can cross the antimeridian or surround
            # a pole, where a lon/lat mean lands on the wrong side of the globe
            coarsened_coords[cluster] = self.spherical_mean(ocean_coords[cluster_mask])
            coarsened_features[:,:, cluster] = ocean_features[:,:, cluster_mask].mean(axis=-1)
        return (coarsened_coords, coarsened_features)# ((num_clusters, 2), (num_features, num_time_steps, num_clusters))
    
    def prepare_inputs_clustering(self, pixels):
        '''prepare the inputs for the clustering algorithm. The clustering is purely geometric:
        an embedding of the lon lat coordinates is used as features, the pixel variables are not.
        pixels: tuple of (coords, features) for the grid cells
        returns: numpy array of shape (num_grid_cells, num_lonlat_embedding)'''
        coords, _ = pixels
        # unit-sphere embedding of coords so that spatial proximity is represented without discontinuities
        return self.unit_sphere_embedding(coords)
    

    def unit_sphere_embedding(self, coords):
        '''create a unit-sphere cartesian embedding of the coordinates
        coords: numpy array of shape (num_grid_cells, 2) with longitude and latitude (in that order)
        returns: numpy array of shape (num_grid_cells, 3) with [x, y, z] on the unit sphere.
        The embedding is injective and chord distance in it is monotone in great-circle distance,
        so a knn graph over it is a spatial knn graph.'''
        assert coords.shape[1] == 2, "Coordinates should have shape (num_grid_cells, 2) with longitude and latitude"
        grid_lon_rad = np.deg2rad(coords[:,0])
        grid_lat_rad = np.deg2rad(coords[:,1])
        grid_features_list = []
        grid_features_list.append(np.cos(grid_lat_rad)*np.cos(grid_lon_rad))
        grid_features_list.append(np.cos(grid_lat_rad)*np.sin(grid_lon_rad))
        grid_features_list.append(np.sin(grid_lat_rad))
        return np.array(grid_features_list).transpose()

    def spherical_mean(self, coords):
        '''mean position of the coordinates on the sphere: the mean of their unit-sphere embedding,
        rescaled onto the sphere. Unlike the mean of lon and lat it does not break for points on
        both sides of the antimeridian or around a pole.
        coords: numpy array of shape (num_points, 2) with longitude and latitude (in that order)
        returns: numpy array of shape (2,) with longitude and latitude of the mean'''
        x, y, z = self.unit_sphere_embedding(coords).mean(axis=0)
        # arctan2 only uses the direction of (x, y, z), which rescales the mean onto the sphere
        return np.rad2deg([np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))])

    def _edge_index_from_knn(self, node_features, k=None):
        '''Build a k-nearest-neighbour graph over the node features.
        node_features: array of shape (num_nodes, num_features), or a list of such arrays to concatenate
        k: number of neighbours, defaults to self.num_neighbours
        returns: edge_index of shape (2, num_edges) with self loops
        The features are used as they are: for the unit-sphere coordinates this makes the graph a
        true spatial knn graph, while standardizing the axes independently would warp the sphere
        into an ellipsoid and the neighbours would no longer be the great-circle nearest ones.'''
        from torch_geometric.nn import knn_graph

        if isinstance(node_features, list):
            node_features = torch.Tensor(np.concatenate(node_features, axis=0))
        else:
            node_features = torch.Tensor(node_features)

        if k is None:
            k = self.num_neighbours

        edge_index = knn_graph(node_features, k=k, loop=True, num_workers=16)
        edge_index = add_remaining_self_loops(edge_index)[0]

        return edge_index

    def get_antarctica_landmask(self):
        """Get a boolean mask for the land grid cells in Antarctica
        Returns:
            np.ndararry of shape (num_land_grid_cells,) with True for land grid cells in Antarctica, False otherwise
        """
        # coordinates hawaii
        hawaii_mask = (self.land[0][:,0] > -161) & (self.land[0][:,0] < -150) & (self.land[0][:,1] > 18) & (self.land[0][:,1] < 23)
        antarctica_mask = self.land[0][:,1]<-60
        # check if the land coordinates are in Antarctica
        # return self.land[0][:,1]<-60
        return (hawaii_mask | antarctica_mask)
    

    def __len__(self):
        return len(self.samples)

class TWSAGraphHeteroData(TWSAGraphDataset):
    '''This class inherits from TWSAGraphDataset and creates a HeteroData object to store the features, edge indices, 
    and edge features for the land, ocean, and mesh nodes. 
    The mesh nodes represent the hydrobasins and ocean clusters, and the edges between the nodes are created based 
    on the spatial relationships between the grid cells and the hydrobasins/ocean clusters. 
    The features for the mesh nodes are initialized as the unit-sphere embedding of their coordinates.'''
    def __init__(self, ds, time_period, ocean_clustering=None,
                 num_ocean_clusters=20,
                 land_clustering=None,
                 target='twsa', delay_time=12, input_seq_len=12, output_seq_len=1, 
                 scaling_mean_std=None,                 
                 time_step=None,
                 k_mesh2mesh=10, 
                 k_mesh2grid=1,
                 positional_vars=None, 
                 land_ocean_mask=None, 
                 features_nanfill=-1.,
                 graph_path=None,
                 train_val_test_data='train',
                 use_ocean_nodes=True,
                 local_ocean_coupling=False,
                 twsa_climatology=None,
                 era5_climatology=None):

        super(TWSAGraphHeteroData, self).__init__(ds, time_period,
                                                 ocean_clustering=ocean_clustering,
                                                 num_ocean_clusters=num_ocean_clusters,
                                                 land_clustering=land_clustering,
                                                 target=target, delay_time=delay_time, input_seq_len=input_seq_len,
                                                 output_seq_len=output_seq_len,
                                                 scaling_mean_std=scaling_mean_std,
                                                 time_step=time_step,
                                                 positional_vars=positional_vars,
                                                 land_ocean_mask=land_ocean_mask, #land_pixel_threshold=land_pixel_threshold,
                                                 features_nanfill=features_nanfill,
                                                 graph_path=graph_path,
                                                 train_val_test_data=train_val_test_data,
                                                 use_ocean_nodes=use_ocean_nodes,
                                                 local_ocean_coupling=local_ocean_coupling,
                                                 twsa_climatology=twsa_climatology,
                                                 era5_climatology=era5_climatology)
        # create the HeteroData object to store features, edge_indices, and edge_features
        self.data = pyg.data.HeteroData()
        self.time_period = time_period
        self.hydro_basins = gpd.read_file('data/orderedhydrosheds/hydrosheds_basins.shp')
        self.land_to_basin_ids = self.get_land_to_basin_ids()
        # get rid of all basins that are not assigned to any land grid cell (usually small islands)
        self.hydro_basins = self.hydro_basins[self.hydro_basins['HYBAS_ID'].isin(self.land_to_basin_ids.unique())].reset_index()
        
       
        self.hydro_basins['basin_centroid'] = self.basin_centroids(self.land[0], self.land_to_basin_ids, self.hydro_basins['HYBAS_ID'])
        
        self.data['grid'].node_index = np.arange(len(self.land[0])) # (num_grid_cells, 1)
        if self.use_ocean_nodes:
            self.data['ocean'].node_index = np.arange(len(self.coarse_ocean[0])) # (num_ocean_clusters, 1)
        self.data['mesh'].node_index = np.arange(len(self.hydro_basins)+len(self.coarse_ocean[0]))
        self.data['mesh'].x = self._init_mesh_features(self.hydro_basins['basin_centroid'].values, self.coarse_ocean[0]) # (num_mesh_cells, 3) unit-sphere embedding
        

        self.data['grid', 'to', 'mesh'].edge_index = self._init_pixelbasin_edgeindex(self.land_to_basin_ids)
        self.data['grid', 'to', 'mesh'].edge_attr = torch.ones(self.data['grid', 'to', 'mesh'].edge_index.shape[1]) # (num_grid_cells, 1)
        if self.use_ocean_nodes:
            self.data['ocean', 'to', 'mesh'].edge_index = torch.from_numpy(np.vstack([self.data['ocean'].node_index, self.data['mesh'].node_index[-self.num_ocean_clusters:]])).long()
            self.data['ocean', 'to', 'mesh'].edge_attr = torch.ones(self.data['ocean', 'to', 'mesh'].edge_index.shape[1]) # (num_ocean_clusters, 1)
        self.data['mesh', 'to', 'mesh'].edge_index, self.data['mesh', 'to', 'mesh'].edge_attr = self._init_basinbasin_edges(k=k_mesh2mesh)
        

        
        self.data['mesh', 'to','grid'].edge_index = self._init_basinpixel_edgeindex(k=k_mesh2grid) 
        # self.draw_graph(edgename='meshgrid', save_dir='meshgrid.png') # plotting is not the nicest yet
        self.data['mesh', 'to', 'grid'].edge_attr = torch.ones(self.data['mesh', 'to', 'grid'].edge_index.shape[1]) # (num_grid_cells, 1)


    def get_land_to_basin_ids(self):
        
        try:
            df = pd.read_csv(self.graph_path+'/landpoints_to_basin.csv') 
            # check if geometry aligns with the coordinates in self.land[0]
            lc = np.array(df[['lon', 'lat']].values)
            assert np.allclose(lc, self.land[0], atol=1e-5), "Coordinates do not match with the hydrobasin coordinates"
            self.land_to_basin_ids = df['HYBAS_ID'].astype('int64')
            log_cache_hit(self.graph_path+'/landpoints_to_basin.csv')

        except (FileNotFoundError, AssertionError):    
            
            self.land_to_basin_ids, joined_df = self.assign_hydrobasins(self.land[0], self.hydro_basins.copy())
            coords = np.array([[geom.x, geom.y] for geom in joined_df['geometry'].values])
            
            joined_df[['lon', 'lat']] = pd.DataFrame(coords, index=joined_df.index)
            joined_df[['lon', 'lat', 'HYBAS_ID']].to_csv(self.graph_path+'/landpoints_to_basin.csv', index=False)
               
        return self.land_to_basin_ids
    
    def assign_hydrobasins(self, coords, gdf):            
            ''' finds the hydrobasin for every grid cell on land
                puts all the not assignable points into the artificial basin 1030099999
                (points in antarctica and hawaii can not be assigned since hydrobasins are not defined there)
            Args: coords: coordinates of the land grid cells
                    gdf: geodataframe of the hydrobasins
            Returns: basin_ids: list of basin IDs for the input coords,
                        joined: GeoDataFrame of land coords (points) and basin ID
            '''
            points = [Point(lon, lat) for lon, lat in coords]
            points_gdf = gpd.GeoDataFrame(geometry=points, crs=gdf.crs)
            
            gdf_split = self.handle_dateline_crossing(gdf)
            # Perform spatial join to find nearest basin for each point
            joined = gpd.sjoin_nearest(points_gdf, gdf_split, how='left', max_distance=0.1)
            # a point equidistant from two basins gets one row per basin, which would make the
            # result longer than the input points - keep the first match so that there is
            # exactly one basin per point
            joined = joined[~joined.index.duplicated(keep='first')]

            # deal with points that could not be assigned (these are in antarctica and hawaii)
            # 1030099999 is the artificial basin they all get - it already exists in
            # hydrosheds_basins.shp as a degenerate polygon at (90, -80), too far from any
            # point for max_distance=0.1 to match, so this fallback is what assigns it
            na_mask = joined['HYBAS_ID'].isna()
            joined.loc[na_mask, 'HYBAS_ID'] = 1030099999

            basin_ids = joined['HYBAS_ID'].astype('int64')
            return basin_ids, joined
    
    def handle_dateline_crossing(self, gdf):
        # Handle dateline crossing by splitting polygons at dateline
        gdf_split = gdf.copy()
        mask_dateline = gdf_split.bounds['maxx'] > 180
        
        # For polygons that cross dateline, split them into two parts
        for idx in gdf_split[mask_dateline].index:
            geom = gdf_split.loc[idx, 'geometry']
            # Split at dateline and wrap coordinates
            west_part = shapely.ops.clip_by_rect(geom, 0, -90, 180, 90)
            true_east_part = shapely.ops.clip_by_rect(geom, -180, -90, 0, 90)
            east_part = shapely.ops.clip_by_rect(geom, 180, -90, 540, 90)
        # Handle potential MultiPolygons from clipping
            if west_part is not None:
                if isinstance(west_part, shapely.MultiPolygon):
                    west_polys = list(west_part.geoms)
                else:
                    west_polys = [west_part]
            else:
                west_polys = []
            
            if true_east_part is not None:
                if isinstance(true_east_part, shapely.MultiPolygon):
                    true_east_polys = list(true_east_part.geoms)
                else:
                    true_east_polys = [true_east_part]
            else:
                true_east_polys = []
                
            if east_part is not None:
                if isinstance(east_part, shapely.MultiPolygon):
                    east_polys = [shapely.ops.transform(
                        lambda x, y: (x - 360 if x >= 180 else x, y), 
                        poly) for poly in east_part.geoms]
                else:
                    east_polys = [shapely.ops.transform(
                        lambda x, y: (x - 360 if x >= 180 else x, y), 
                        east_part)]
            else:
                east_polys = []
            
            # Combine all parts into a MultiPolygon
            all_polys = west_polys + east_polys + true_east_polys
            if all_polys:
                gdf_split.loc[idx, 'geometry'] = shapely.MultiPolygon(all_polys)
        return gdf_split

    def basin_centroids(self, land_coords, land_to_basin_ids, basin_ids):
        '''position of every basin: the spherical mean of the land grid cells assigned to it
        Args: land_coords: numpy array of shape (num_land_grid_cells, 2) with longitude and latitude
              land_to_basin_ids: basin ID of every land grid cell
              basin_ids: basin IDs in mesh node order, each assigned to at least one land grid cell
        Returns: list of shapely Points (lon, lat), one per basin'''
        land_to_basin_ids = np.asarray(land_to_basin_ids)
        return [Point(*self.spherical_mean(land_coords[land_to_basin_ids == bid])) for bid in basin_ids]

    def get_mesh_coords(self):
        '''Get the coordinates of the mesh nodes'''
        hydro_basin_coords = np.array([[geom.x, geom.y] for geom in self.hydro_basins['basin_centroid'].values])
        
        return np.concatenate([hydro_basin_coords,self.coarse_ocean[0]], axis=0)
    
    def _init_mesh_features(self, hydro_basin_centroids, ocean_coords):
        '''Initialize mesh features as the lon and lat of the basin/ocean cluster'''
        hydro_basin_coords = np.array([[geom.x, geom.y] for geom in hydro_basin_centroids])
        xyz_basin = self.unit_sphere_embedding(hydro_basin_coords)
        xyz_ocean = self.unit_sphere_embedding(ocean_coords)

        return torch.Tensor(np.concatenate([xyz_basin, xyz_ocean], axis=0))
   
    def _init_pixelbasin_edgeindex(self, land_to_basin_ids):
        """
        Create edge index connecting points to their basins + connecting ocean nodes to ocean nodes (1 to 1 mapping).
        Some basins have no grid points assinged to them. Some grid points are not assigned to any basin (e.g. in Antarctica).
        For not assignable grid points we assign them to an artificial basin.
        Returns:
            edge_index: torch.Tensor with shape (2, num_edges) containing [point_idx, basin_idx] pairs
        """
        try: 
            with open(self.graph_path+"/pixel_basin_edgeindex.pt", "rb") as f:
                edge_index = torch.load(f, weights_only=True)
            log_cache_hit(self.graph_path+"/pixel_basin_edgeindex.pt")
        except (FileNotFoundError, TypeError):
            print("Calculating edge index for pixel to basin mapping")
            # Get point indices and basin indices
            point_indices = self.data['grid'].node_index  # Index of each point
            # Create mapping from HYBAS_ID to position index
            basin_id_to_index = {row.HYBAS_ID: idx for idx, row in self.hydro_basins.iterrows()}
            
            # Map each land point's basin ID to its index
            basin_indices = np.array([basin_id_to_index[bid] for bid in land_to_basin_ids])
        
            # Stack into edge index format (2, num_edges)
            edge_index = np.vstack([point_indices, basin_indices])
            
            # Convert to torch tensor
            edge_index = torch.from_numpy(edge_index).long()
            # Save edge index to file
            torch.save(edge_index, self.graph_path+"/pixel_basin_edgeindex.pt")
        
        return edge_index


    
    def _init_basinbasin_edges(self, k):
        ''' Edge Index for intersecting basins loaded from file or calculated from knn algorithm.        
        '''
        try: 
            with open(self.graph_path+f"/basin_basin_edgeindex{k}.pt", "rb") as f:
                edgeindex = torch.load(f, weights_only=True)
            log_cache_hit(self.graph_path+f"/basin_basin_edgeindex{k}.pt")
        except (FileNotFoundError, TypeError):
            print("Calculating edge index for basin to basin mapping")
            
            if k==0:
                # fully connected mesh graph 
                edgeindex = torch.tensor([[i, j] for i in range(self.data['mesh'].x.shape[0]) for j in range(self.data['mesh'].x.shape[0])]).t()
            else:
                # calculate the edge index from the knn graph of the basin centroids
                xyz_coords = self.data['mesh'].x.numpy()
                edgeindex = self._edge_index_from_knn(xyz_coords,k)
                if self.use_ocean_nodes:
                    edgeindex = torch.cat([edgeindex, self._ocean_mesh_edges()], dim=1)
            torch.save(edgeindex, self.graph_path+f"/basin_basin_edgeindex{k}.pt")
        edge_attr = torch.ones(edgeindex.shape[1])
        return edgeindex, edge_attr

    def _ocean_mesh_edges(self, num_nearest=3):
        '''Outgoing edges of the ocean cluster nodes in the mesh to mesh graph, with the ocean
        node as source (edge_index[0]).
        full variant: every ocean node sends to every mesh node.
        local_ocean variant: each basin only receives from its num_nearest closest ocean clusters,
        no ocean to ocean edges are added (only those of the knn graph remain). The mesh features are the unit-sphere
        embedding, where chord distance is monotone in great-circle distance, so the knn over
        them gives the great-circle nearest clusters.'''
        num_mesh = self.data['mesh'].x.shape[0]
        first_ocean = num_mesh - len(self.coarse_ocean[0])
        ocean_idx = np.arange(first_ocean, num_mesh)

        if not self.local_ocean_coupling:
            return torch.tensor([[i, j] for i in ocean_idx for j in range(num_mesh)]).t()

        # ocean_ocean = torch.tensor([[i, j] for i in ocean_idx for j in ocean_idx]).t() # connects all ocean nodes to each other
        xyz = self.data['mesh'].x
        ocean_basin = pyg.nn.knn(xyz[first_ocean:], xyz[:first_ocean], num_nearest).flip(0)
        ocean_basin[0] += first_ocean # ocean cluster index to mesh index, row 1 is already the basin index
        return ocean_basin


    def _init_basinpixel_edgeindex(self, k=5):
        ''' Mapping back from the basins to the grid cells. 
        Includes grid cells that are not assignable to any basin and ocean grid cells.'''
       
        try: 
            with open(self.graph_path+f"/basin_pixel_edgeindex_knn{k}.pt", "rb") as f:
                edge_index = torch.load(f, weights_only=True)
            log_cache_hit(self.graph_path+f"/basin_pixel_edgeindex_knn{k}.pt")
        except (FileNotFoundError, TypeError):
            print("Calculating edge index for basin to pixel mapping")
            xyz_coords_mesh = self.data['mesh'].x.numpy()
            xyz_grid_coords = self.unit_sphere_embedding(self.land[0])
            edge_index = pyg.nn.knn(torch.from_numpy(xyz_coords_mesh[:len(self.hydro_basins)]).to(torch.float32), torch.from_numpy(xyz_grid_coords).to(torch.float32), k).flip(0)
            torch.save(edge_index, self.graph_path+f"/basin_pixel_edgeindex_knn{k}.pt")
        return edge_index
        
    
    
   
    def _get_greenland_basins(self):
        basin_id_list = [9030013780,9030000010,	9030010370,	9030005280]
        return basin_id_list
    def get_greenland_landmask(self):
        """Get a boolean mask for the land grid cells in Greenland
        Returns:
            np.ndararry of shape (num_land_grid_cells,) with True for land grid cells in Greenland, False otherwise
        """
        greenland_basins = self._get_greenland_basins()
        mask = self.land_to_basin_ids.isin(greenland_basins)
        return mask.values

    def num_land_nodes(self):
        return len(self.land[0])
    def num_nodes(self):
        return len(self.land[0]) + len(self.coarse_ocean[0])
    
    

    def __getitem__(self, idx):
        # shallow copy: the sample gets its own attribute dicts but shares the edges and mesh
        # positions with self.data, which are identical for every sample. Safe because the
        # sample only rebinds x/y and no consumer writes into a graph tensor in place.
        data_chunk = copy.copy(self.data)

        # get sample
        t = self.samples[idx]
        # get features
        input_seq = self.features[:,t-self.seq_len:t] # shape  [grid cells (land), num_time_steps, num_variables]
        # get target
        pred_start = t + self.delay
        pred_end = t + self.delay + self.pred_len
        target_seq = self.target_series[:, pred_start: pred_end] # shape [grid cells (land), num_time_steps]

        data_chunk['grid'].x = input_seq
        if self.use_ocean_nodes:
            data_chunk['ocean'].x = self.ocean_features[:,t-self.seq_len:t] # shape  [num_ocean_clusters, num_time_steps,  num_variables]
        data_chunk['grid'].y = target_seq
        if self.twsa_offset is not None:
            data_chunk['grid'].twsa_offset = self.twsa_offset[:, pred_start: pred_end] # C of the target months, added back in predict_step

        return data_chunk
    



class DummyDataset(TWSAGraphHeteroData):
    ''' Dummmy dataset for testing and debugging.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        data_chunk = super().__getitem__(idx)
        data_chunk['grid'].x = torch.ones(data_chunk['grid'].x.shape)
        if self.use_ocean_nodes:
            data_chunk['ocean'].x = torch.ones(data_chunk['ocean'].x.shape)

        return data_chunk


