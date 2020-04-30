#!/usr/bin/env python

import pathlib
import glob
import numpy as np


print("Loading input data...")

data_dir = pathlib.Path('input_data')

imputed_anom_3D = np.load(data_dir / "imputed_anom_3D-v2.npy")

master_mask_2D = np.load(data_dir / "master_mask_2D-v2.npy")
nan_mask_3D = np.load(data_dir / "nan_mask_3D-v2.npy")
evaluation_mask_3D = np.load(data_dir / "evaluation_mask_3D-v2.npy")

duration, height, width = imputed_anom_3D.shape

row_idxs = np.load(data_dir / "row_idxs-v2.npy")
col_idxs = np.load(data_dir / "col_idxs-v2.npy")


def col_idx_shift(row, col):
    new_col = col-(row - 12)//2+3
    return row, new_col
shifted_width = 96

print("Shifting colon indexes...")
shifted_col_idxs = [col_idx_shift(row, col)[1] for row, col in zip(row_idxs, col_idxs)]
np.save(data_dir / "shifted_col_idxs-v2", shifted_col_idxs)


def shift_2D_bitmap(bitmap):
    assert bitmap.shape == (height, width)
    shifted_bitmap = np.zeros((height, shifted_width), dtype=bitmap.dtype)
    shifted_bitmap[row_idxs, shifted_col_idxs] = bitmap[row_idxs, col_idxs]
    return shifted_bitmap

def shift_3D_bitmap(bitmap):
    assert bitmap.shape[1:] == (height, width)
    shifted_bitmap = np.zeros((bitmap.shape[0], height, shifted_width), dtype=bitmap.dtype)
    shifted_bitmap[:, row_idxs, shifted_col_idxs] = bitmap[:, row_idxs, col_idxs]
    return shifted_bitmap


print("Shifting master mask...")
master_mask_2D_shifted = shift_2D_bitmap(master_mask_2D)
np.save(data_dir / "master_mask_2D_shifted-v2", master_mask_2D_shifted)


print("Shifting training masks...")
base_input_file_name = "training_mask_3D-v2"
base_output_file_name = "training_mask_3D_shifted-v2"
for input_file_name in glob.glob(str(data_dir / (base_input_file_name + "*"))):
    print("..." + input_file_name)
    input_file_name_suffix_position = input_file_name.find("training_mask_3D-v2") + len(base_input_file_name)
    input_file_name_suffix = str(pathlib.Path(input_file_name[input_file_name_suffix_position:]).stem)
    output_file_name = data_dir / (base_output_file_name + input_file_name_suffix)
    training_mask_3D = np.load(input_file_name)
    training_mask_3D_shifted = shift_3D_bitmap(training_mask_3D)
    np.save(output_file_name, training_mask_3D_shifted)


print("Shifting NaN mask...")
nan_mask_3D_shifted = shift_3D_bitmap(nan_mask_3D)
np.save(data_dir / "nan_mask_3D_shifted-v2", nan_mask_3D_shifted)


print("Shifting imputed anomaly...")
imputed_anom_3D_shifted = shift_3D_bitmap(imputed_anom_3D)
np.save(data_dir / "imputed_anom_3D_shifted-v2", imputed_anom_3D_shifted)


print("Shifting evaluation mask...")
evaluation_mask_3D_shifted = shift_3D_bitmap(evaluation_mask_3D)
np.save(data_dir / "evaluation_mask_3D_shifted-v2", evaluation_mask_3D_shifted)