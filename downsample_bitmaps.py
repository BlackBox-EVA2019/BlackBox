#!/usr/bin/env python

# Downsampling is done on shifted anomaly and maps

import pathlib
import numpy as np
import glob
import parse
from tqdm import tqdm  # for progress bars

converted_data_dir = pathlib.Path('input_data')

master_mask_2D_shifted = np.load(converted_data_dir / "master_mask_2D_shifted-v2.npy")

evaluation_mask_3D_shifted = np.load(converted_data_dir / "evaluation_mask_3D_shifted-v2.npy")
(duration, height, width) = evaluation_mask_3D_shifted.shape

# Integer factor for downsampling
factor = 3 # change this and rerun, file names are uniquely generated

assert(width % factor == 0)
new_width = width // factor
assert(height % factor == 0)
new_height = height // factor

#Downsampling master mask
master_mask_2D_shifted_ds = np.zeros((new_height, new_width), dtype=np.float32)
for row in range(new_height):
    for col in range(new_width):
        master_mask_2D_shifted_ds.itemset((row, col),
        master_mask_2D_shifted[row*factor:(row+1)*factor,col*factor:(col+1)*factor].astype(np.float32).sum() / (factor**2))
        
fname = "master_mask_2D_shifted-v2_ds_" + str(factor) + "x" + str(factor)
np.save(converted_data_dir / fname, master_mask_2D_shifted_ds)

nan_mask_3D_shifted = np.load(converted_data_dir / "nan_mask_3D_shifted-v2.npy")
imputed_anom_3D_shifted = np.load(converted_data_dir / "imputed_anom_3D_shifted-v2.npy")

#Downsampling evaluation mask
evaluation_mask_3D_shifted_ds = np.zeros((duration, new_height, new_width), dtype=np.float32)
imputed_anom_3D_shifted_ds_on_em = np.zeros((duration, new_height, new_width), dtype=np.float32)

print("Downsampling evaluation data...")
for t, eval_mask_shifted in tqdm(enumerate(evaluation_mask_3D_shifted), total=duration):
    assert((eval_mask_shifted & ~master_mask_2D_shifted).all()==False)
    for row in range(new_height):
        for col in range(new_width):
            num_of_cells = eval_mask_shifted[row*factor:(row+1)*factor, col*factor:(col+1)*factor].sum()
            evaluation_mask_3D_shifted_ds.itemset((t, row, col), num_of_cells / (factor ** 2))
            if(num_of_cells > 0):
                imputed_anom_3D_shifted_ds_on_em.itemset((t, row, col),
                                imputed_anom_3D_shifted[t, row*factor:(row+1)*factor, col*factor:(col+1)*factor].sum() /
                                num_of_cells)

fname = converted_data_dir / ("evaluation_mask_3D_shifted-v2_ds_" + str(factor) + "x" + str(factor))
np.save(fname, evaluation_mask_3D_shifted_ds)

fname = converted_data_dir / ("imputed_anom_3D_shifted-v2_ds_" + str(factor) + "x" + str(factor))
np.save(fname, imputed_anom_3D_shifted_ds_on_em)

#Downsampling nan mask
nan_mask_3D_shifted_ds = np.zeros((duration, new_height, new_width), dtype=np.float32)

print("Downsampling nan mask...")
for t, nan_mask_shifted in tqdm(enumerate(nan_mask_3D_shifted), total=duration):
    assert((nan_mask_shifted & ~master_mask_2D_shifted).all()==False)
    for row in range(new_height):
        for col in range(new_width):
            num_of_cells = nan_mask_shifted[row*factor:(row+1)*factor, col*factor:(col+1)*factor].sum()
            nan_mask_3D_shifted_ds.itemset((t, row, col), num_of_cells / (factor ** 2))

fname = "nan_mask_3D_shifted-v2_ds_" + str(factor) + "x" + str(factor)
np.save(converted_data_dir / fname, nan_mask_3D_shifted_ds)

#Downsampling multiple training and validation masks
print("Downsampling multiple training and validation masks...")
base_training_mask_file_name = "training_mask_3D_shifted-v2"
for training_mask_file_name in glob.glob(str(converted_data_dir / (base_training_mask_file_name + "*"))):
    training_mask_pathlib_name = pathlib.Path(training_mask_file_name).stem
    
    # Skips training masks that are allready downsampled
    if parse.parse(base_training_mask_file_name + "-{N1:g}-{N2:g}", training_mask_pathlib_name) == None :
        continue
    
    print("  Working on ", training_mask_pathlib_name, "...")
    
    training_mask_3D_shifted = np.load(training_mask_file_name)
    
    # Downsampling training mask
    training_mask_3D_shifted_ds = np.zeros((duration, new_height, new_width), dtype=np.float32)
    imputed_anom_3D_shifted_ds_on_tm = np.zeros((duration, new_height, new_width), dtype=np.float32)

    for t, train_mask_shifted in tqdm(enumerate(training_mask_3D_shifted), total=duration):
        assert((train_mask_shifted & ~master_mask_2D_shifted).all()==False)
        for row in range(new_height):
            for col in range(new_width):
                num_of_cells = train_mask_shifted[row*factor:(row+1)*factor, col*factor:(col+1)*factor].sum()
                training_mask_3D_shifted_ds.itemset((t, row, col), num_of_cells / (factor ** 2))
                if(num_of_cells > 0):
                    imputed_anom_3D_shifted_ds_on_tm.itemset((t, row, col),
                                    imputed_anom_3D_shifted[t, row*factor:(row+1)*factor, col*factor:(col+1)*factor].sum() /
                                    num_of_cells)
    
    assert not np.isnan(imputed_anom_3D_shifted_ds_on_tm).any()
    
    fname = converted_data_dir / (str(training_mask_pathlib_name) + "_ds_" + str(factor) + "x" + str(factor))
    np.save(fname, training_mask_3D_shifted_ds)
    
   
    # Downsampling validation mask
    validation_mask_3D_shifted_ds = np.zeros((duration, new_height, new_width), dtype=np.float32)
    imputed_anom_3D_shifted_ds_on_vm = np.zeros((duration, new_height, new_width), dtype=np.float32)

    for t, valid_mask_shifted in tqdm(enumerate(evaluation_mask_3D_shifted & ~training_mask_3D_shifted), total=duration):
        assert((valid_mask_shifted & ~master_mask_2D_shifted).all()==False)
        for row in range(new_height):
            for col in range(new_width):
                num_of_cells = valid_mask_shifted[row*factor:(row+1)*factor, col*factor:(col+1)*factor].sum()
                validation_mask_3D_shifted_ds.itemset((t, row, col), num_of_cells / (factor ** 2))
                if(num_of_cells > 0):
                    imputed_anom_3D_shifted_ds_on_vm.itemset((t, row, col),
                                    imputed_anom_3D_shifted[t, row*factor:(row+1)*factor, col*factor:(col+1)*factor].sum() /
                                    num_of_cells)

    assert not np.isnan(imputed_anom_3D_shifted_ds_on_vm).any()
    
    fname = converted_data_dir / ("validation_mask_3D_shifted-v2_ds_" + str(factor) + "x" + str(factor) + "_from_" + str(training_mask_pathlib_name))
    np.save(fname, validation_mask_3D_shifted_ds)
    