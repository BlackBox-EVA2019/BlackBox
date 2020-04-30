#!/usr/bin/env python

import os
import pathlib
import numpy as np
from tqdm import tqdm
from functools import partial

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
numpy2ri.activate()
fields = importr("fields")

make_pbar = partial(tqdm, leave=False)


converted_data_dir = pathlib.Path('input_data')
try:
    os.mkdir(converted_data_dir)
except FileExistsError:
    pass


print("Loading R data...")
robj = robjects.r.load('DATA_TRAINING.RData')
data = {name: np.array(robjects.r[name]) for name in robj}
anom = data["anom.training"]
loc = data["loc"]


print("Calculating distances...")
space_len = loc.shape[0]
dist = np.zeros((space_len, space_len))
for j, l in enumerate(make_pbar(loc)):
    dist[j] = np.array(fields.rdist_earth(x1=l[np.newaxis, :], x2=loc, miles=False))
np.save(converted_data_dir / "dist", dist)


def get_bitmap_idxs(loc):
    '''map 1D space locations to 2D [height, width] indices'''
    lon, lat = loc.T

    # reconstruct bitmap grid
    lon_unique, lat_unique = sorted(list(set(lon))), sorted(list(set(lat)))
    lat_unique.reverse()
    #print("Unique longitudes: ", len(lon_unique))
    #print("Unique latitudes:  ", len(lat_unique))
    
    # adjust the grid to 384 x 256 (3*128 x 2*128)
    lon_to_col_idx = {l: x + 12 for x, l in enumerate(lon_unique)}
    lat_to_row_idx = {l: y + 12 for y, l in enumerate(lat_unique)}

    col_idxs = [lon_to_col_idx[_lon] for _lon in lon]
    row_idxs = [lat_to_row_idx[_lat] for _lat in lat]
    
    return row_idxs, col_idxs

print("Preparing bitmap indices...")
row_idxs, col_idxs = get_bitmap_idxs(loc)

np.save(converted_data_dir / "row_idxs-v2", row_idxs)
np.save(converted_data_dir / "col_idxs-v2", col_idxs)

#def col_idx_shift(row, col):
#    new_col = col-(row - 12)//2+3
#    return row, new_col

#shifted_col_idxs = [col_idx_shift(row, col)[1] for row, col in zip(row_idxs, col_idxs)]

#np.save(converted_data_dir / "shifted_col_idxs-v2", shifted_col_idxs)

width = 256
height = 384

print("Creating lookup index...")
lookup_original_idx_2D = np.full((height, width), -1, dtype=np.int32)
for original_idx, (row, col) in enumerate(zip(row_idxs, col_idxs)):
    lookup_original_idx_2D.itemset((row, col), original_idx)
np.save(converted_data_dir / "lookup_original_idx_2D-v2", lookup_original_idx_2D)
      

print("Creating master mask bitmap...")
master_mask_2D = np.zeros((height, width), dtype=bool)
master_mask_2D[row_idxs, col_idxs] = True
np.save(converted_data_dir / "master_mask_2D-v2", master_mask_2D)


print("Creating anomaly bitmap...")
duration = len(anom)

anom_3D = np.zeros((duration, height, width), dtype=np.float32)
for bitmap_aspace, aspace in zip(anom_3D, make_pbar(anom)):
    bitmap_aspace[row_idxs, col_idxs] = aspace
#np.save(converted_data_dir / "anom_3D-v2", anom_3D)


print("Creating NaN mask bitmap...")
nan_mask_3D = np.isnan(anom_3D)
np.save(converted_data_dir / "nan_mask_3D-v2", nan_mask_3D)


print("Creating zero-imputed anomaly bitmap...")
#imputed_anom_3D = np.copy(anom_3D)
imputed_anom_3D = anom_3D
imputed_anom_3D[nan_mask_3D] = 0
np.save(converted_data_dir / "imputed_anom_3D-v2", imputed_anom_3D)


print("Creating evaluation mask bitmap...")
evaluation_mask_3D = ~nan_mask_3D & master_mask_2D
np.save(converted_data_dir / "evaluation_mask_3D-v2", evaluation_mask_3D)
