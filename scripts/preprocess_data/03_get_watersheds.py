import ee
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
from pyproj import Transformer
from shapely.geometry import shape

'''
This script uses the Google Earth Engine API to access the HydroSHEDS dataset and extract watershed basin information. 
It creates an adjacency matrix based on shared borders between basins and saves the results in a shapefile.
The script also includes functions to find neighboring basins and plot a basin with its neighbors.

'''
def find_basin_for_point(lat, lon, dataset):
    """Find the watershed basin that contains the given point.
    # """
    # Create a point geometry
    point = ee.Geometry.Point([lon, lat])
    
    # Filter the dataset to find basins that contain the point
    basin = dataset.filter(ee.Filter.bounds(point)).first()
    
    return basin

def get_points_in_basins(points_list, dataset):
    """
    Assign multiple points to their respective basins
    points_list: List of [lat, lon] coordinates
    """
    results = []
    for lat, lon in points_list:
        
        basin = find_basin_for_point(lat, lon, dataset)
        
        # Get basin properties
        basin_info = {
            'point': [lat, lon],
            'basin_id': basin.get('HYBAS_ID'),
            'basin_area': basin.get('SUB_AREA'),
            'basin_geometry': basin.geometry()
        }
        results.append(basin_info)
    
    return results

def find_neighboring_basins(basin_id, dataset):
    """
    Find all basins that share a border with the given basin
    Args:
        basin_id: HYBAS_ID of the target basin
        dataset: HydroSHEDS feature collection
    Returns:
        List of neighboring basin IDs
    """
    # Get the target basin
    target_basin = dataset.filter(ee.Filter.eq('HYBAS_ID', basin_id)).first()
    target_geometry = target_basin.geometry()
    
    # Find intersecting basins    
    neighbors = dataset.filter(
        ee.Filter.And(
            ee.Filter.intersects('.geo', target_geometry),
            # ee.Filter.neq('HYBAS_ID', basin_id) # We exclude the basin itself from the results
        )
    )
    
    return neighbors

def calulate_basin_centroids(dataset):
     # Get all basin features with centroids
    features = dataset.getInfo()['features']
    centroids = {}
    for f in features:
        props = f['properties']
        geom = f['geometry']
        basin_id = props['HYBAS_ID']
        # Calculate centroid if not already in properties
        if 'centroid_lon' not in props:
            
            centroid = shape(geom).centroid
            centroids[basin_id] = np.array([centroid.x, centroid.y])
        else:
            centroids[basin_id] = np.array([props['centroid_lon'], props['centroid_lat']])
    return centroids


def calculate_spherical_distance(c1, c2):
    """Calculate great circle distance between two points"""
    from pyproj import Geod
    
    geod = Geod(ellps='WGS84')
    
    # Handle dateline crossing
    lon1, lat1 = c1[0], c1[1]
    lon2, lat2 = c2[0], c2[1]
    
    # If points might be across dateline, try both ways around
    if abs(lon1 - lon2) > 180:
        # Try regular way
        dist1 = geod.inv(lon1, lat1, lon2, lat2)[2]
        # Try with shifted longitude
        lon2_shift = lon2 - 360 if lon2 > 0 else lon2 + 360
        dist2 = geod.inv(lon1, lat1, lon2_shift, lat2)[2]
        # Take shorter distance
        return min(dist1, dist2) / 1000.0  # Convert to km
    else:
        return geod.inv(lon1, lat1, lon2, lat2)[2] / 1000.0  # Convert to km



def create_adjacency_matrix(dataset):
    """
    
    Create an adjacency matrix for HydroSHEDS basins where 1 indicates shared borders
    and 0 indicates no shared border.
    
    Args:
        dataset: ee.FeatureCollection of HydroSHEDS basins
    
    Returns:
        numpy array: Adjacency matrix where rows and columns represent basin IDs
        dict: Mapping of basin IDs to matrix indices
    """

    # Sort basins by HYBAS_ID to ensure consistent ordering
    # then get all basin IDs
    basins = dataset.sort('HYBAS_ID').aggregate_array('HYBAS_ID').getInfo()
    n_basins = len(basins)
    
    # Create a mapping of basin IDs to matrix indices
    basin_to_index = {basin_id: idx for idx, basin_id in enumerate(basins)}
    # Create a mapping of matrix indices to basin IDs
    index_to_basin = {idx: basin_id for idx, basin_id in enumerate(basins)}
    
    # Initialize adjacency matrix with zeros
    adj_matrix = np.zeros((n_basins, n_basins))
    
    neighbors_list = []
    # Iterate through each basin to find its neighbors
    for basin_id in tqdm(basins, desc="Processing basins", unit="basin"):
        neighbors = find_neighboring_basins(basin_id, dataset)
        neighbor_ids = neighbors.aggregate_array('HYBAS_ID').getInfo()
        neighbors_list.append(neighbor_ids)

        # Set 1 for each neighboring relationship
        basin_idx = basin_to_index[basin_id]
        for neighbor_id in neighbor_ids:
            neighbor_idx = basin_to_index[neighbor_id]
            adj_matrix[basin_idx, neighbor_idx] = 1
            adj_matrix[neighbor_idx, basin_idx] = 1  # Matrix is symmetric


    

    return adj_matrix, index_to_basin, basin_to_index, neighbors_list

def normalize_adj_matrix(adj_matrix):
    """
    Symmetric normalization: D^(-1/2) * A * D^(-1/2)
    where D is the degree matrix and A is the adjacency matrix
    """
    #adj_matrix = adj_matrix +np.eye(adj_matrix.shape[0])  # Reinforce self-loops
    # Calculate degree matrix
    D = np.sum(adj_matrix, axis=1)
    # Calculate D^(-1/2)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
    
    # Convert to diagonal matrix
    D_inv_sqrt = np.diag(D_inv_sqrt)
    
    # Symmetric normalization
    normalized_adj = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    
    return normalized_adj

def plot_basin_and_neighbors(basin_id, adj_matrix, idx_to_basin, dataset):
    """
    Plot a basin and its neighbors using the adjacency matrix
    
    Args:
        basin_id: HYBAS_ID of the target basin
        adj_matrix: Adjacency matrix
        idx_to_basin: Dictionary mapping matrix indices to basin IDs
        dataset: HydroSHEDS feature collection
    """

    
    # Find the matrix index for the target basin
    basin_idx = [idx for idx, bid in idx_to_basin.items() if str(bid) == str(basin_id)][0]
    
    # Get neighbor indices from adjacency matrix
    neighbor_indices = np.where(adj_matrix[basin_idx] == 1)[0]
    neighbor_ids = [idx_to_basin[idx] for idx in neighbor_indices]
    
    # Convert target basin and neighbors to GeoDataFrame
    target = dataset.filter(ee.Filter.eq('HYBAS_ID', basin_id))
    neighbors = dataset.filter(ee.Filter.inList('HYBAS_ID', neighbor_ids))
    
    # Combine and get as GeoJSON
    all_features = target.merge(neighbors).getInfo()
    gdf = gpd.GeoDataFrame.from_features(all_features['features'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot neighbors in light blue
    gdf[gdf.HYBAS_ID.isin(neighbor_ids)].plot(ax=ax, color='lightblue', alpha=0.5)
    
    # Plot target basin in red
    gdf[gdf.HYBAS_ID == basin_id].plot(ax=ax, color='red', alpha=0.5)
    
    plt.title(f'Basin {basin_id} and its neighbors')
    # plt.axis('equal')
    # plt.show()
    plt.savefig(f'data/orderedhydrosheds/basin_{basin_id}_neighbors.png')
    plt.close()


def main():

    # ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    ee.Authenticate() ## if this fails then command line: earthengine authenticate --auth_mode=notebook
    ee.Initialize()

    dataset = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_3")

    adj_matrix, idx_to_basin, basin_to_idx, neighbors_list = create_adjacency_matrix(dataset)


    # To save the adjacency matrix
    np.save('data/orderedhydrosheds/basin_adjacency_matrix_intersections.npy', adj_matrix)

    example_basin_id = list(idx_to_basin.values())[100]
    plot_basin_and_neighbors(example_basin_id, adj_matrix, idx_to_basin, dataset)

    # Get all features as GeoJSON
    print("Fetching basins from Earth Engine...")
    features = dataset.getInfo()

    # Convert to GeoDataFrame
    print("Converting to GeoDataFrame...")
    gdf = gpd.GeoDataFrame.from_features(features['features'])
    # neighbors_list is indexed by sorted HYBAS_ID order (basin_to_idx), while the rows of
    # gdf come back in Earth Engine's natural order - join on HYBAS_ID, not on row position
    neighbors_by_id = {basin_id: neighbors_list[idx] for basin_id, idx in basin_to_idx.items()}
    gdf['Neigh_HYBAS_ID'] = gdf['HYBAS_ID'].map(neighbors_by_id)

    # Save to shapefile after sorting GeoDataFrame by HYBAS_ID
    gdf = gdf.sort_values('HYBAS_ID').reset_index(drop=True)
    print("Saving to shapefile...")
    gdf.to_file('data/orderedhydrosheds/hydrosheds_basins.shp')
    # Save to GeoPackage
    print("Saving to GeoPackage...")
    gdf.to_file('data/orderedhydrosheds/hydrosheds_basins.gpkg', driver='GPKG')





if __name__ == "__main__":
    main()