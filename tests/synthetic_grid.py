"""A small synthetic grid shared by the 2D and heterograph normalization tests.

The land column carries the two edge cases the std floor has to handle:
a pixel whose variance is far below the median, and a pixel whose raw values
go negative (runoff).
"""
import numpy as np
import xarray as xr

T = 36
LAT = np.array([-70.5, -59.5, 10.5, 40.5], dtype=np.float32)
LON = np.array([0.5, 1.5, 2.5], dtype=np.float32)

# column 0 is land (lsm > 0.8), column 1 is coastal (0 < lsm <= 0.8), column 2 is ocean
LSM = np.array([[1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [0.9, 0.3, 0.0]], dtype=np.float32)

LAND = LSM > 0.8
COASTAL = (LSM > 0) & (LSM <= 0.8)
OCEAN = LSM == 0

LOW_VARIANCE_CELL = (2, 0)   # std far below median(std) -> must hit the floor
NEGATIVE_RUNOFF_CELL = (3, 0)  # detrended runoff goes negative -> must not be clipped
ORDINARY_CELL = (0, 0)

METEO_VARS = ['twsa', 'sst', 'ro', 't2m']


def season(t, amplitude, offset=0.0, phase=0.0):
    return offset + amplitude * np.sin(2 * np.pi * (np.arange(t) / 12.0) + phase)


def make_ds():
    time = np.arange('2000-01', '2003-01', dtype='datetime64[M]').astype('datetime64[ns]')
    assert len(time) == T
    shape = (T, len(LAT), len(LON))

    # sst: a distinct, time-varying series in every cell, so a cell that survives
    # normalization has non-zero variance and a masked-out cell is exactly zero
    sst = np.empty(shape, dtype=np.float32)
    twsa = np.empty(shape, dtype=np.float32)
    for i in range(len(LAT)):
        for j in range(len(LON)):
            sst[:, i, j] = season(T, amplitude=1.0 + i, offset=280.0 + j, phase=0.3 * (i + j))
            twsa[:, i, j] = season(T, amplitude=2.0, offset=0.0, phase=0.1 * (i + j))

    # ro is a land-only variable; the land column carries the two edge cases
    ro = np.zeros(shape, dtype=np.float32)
    ro[:, 0, 0] = season(T, amplitude=1.0, offset=1.0)
    ro[:, 1, 0] = season(T, amplitude=1.0, offset=2.0, phase=np.pi / 2)
    ro[:, 2, 0] = season(T, amplitude=0.001, offset=5.0)
    ro[:, 3, 0] = season(T, amplitude=5.0, offset=0.0)

    t2m = np.broadcast_to(season(T, 3.0, 290.0)[:, None, None], shape).astype(np.float32)
    lsm = np.broadcast_to(LSM[None, :, :], shape).astype(np.float32)

    dims = ('time', 'lat', 'lon')
    return xr.Dataset(
        {'twsa': (dims, twsa), 'sst': (dims, sst), 'ro': (dims, ro),
         't2m': (dims, t2m), 'lsm': (dims, lsm)},
        coords={'time': time, 'lat': LAT, 'lon': LON},
    )


def expected_runoff_floor():
    """median(std) * 0.01 over the land cells, computed on unclipped runoff."""
    std = make_ds()['ro'].values.astype(np.float32).std(axis=0)
    return float(np.median(std[LAND]) * 0.01)
