#!/usr/bin/env python
# coding: utf-8


import math  # for math.floor()
import pathlib

import numpy as np

from fastai.imports import *
from fastai.basics import *
from fastai.callbacks import *

from torch.utils.data import Subset


# Global variable (for add_noise_in holes() to be fast)
master_mask_2D = None  # should be initialized later


# Preparing datasets for neural networks

class MyNDayDataset(TensorDataset):
    def __init__(self, window_days=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_days = window_days
        self.window_start = - (window_days // 2)
        self.window_end = window_days - (window_days // 2)

    def __getitem__(self, idx):
        *_, Y, Ymask = super().__getitem__(idx)

        window = (super(MyNDayDataset, self).__getitem__(i)
                  for i in range(idx + self.window_start, idx + self.window_end))

        Xs, Xmasks, days, *_ = map(torch.stack, zip(*window))

        return (Xs, Xmasks, days), (Y, Ymask)


# Noise is added to the holes during model training and evaluation.
# Noise profile is hardcoded and precalculated from existing data
def add_noise_in_holes(input_tensors):
    global master_mask_2D
    (Xs, Xmasks, days), (Y, Ymask) = input_tensors
    noisy_Xs = torch.empty(Xs.shape, device=Xs.device).normal_(
        0, 0.46647381**0.5 / 4)
    noisy_Xs.mul_(1 - Xmasks).mul_(master_mask_2D.to(Xs.device)
                                   ).add_(Xs * Xmasks)
    return (noisy_Xs, Xmasks, days), (Y, Ymask)


# Load data

converted_data_dir = pathlib.Path('input_data')

day_d = 8030  # Day when the big holes start


def get_validation_set_position(duration, val_set_percent):
    duration_valid_set = round(duration * val_set_percent)
    valid_set_day_start = day_d - round(duration_valid_set * (22 / 31))
    valid_set_day_end = day_d + round(duration_valid_set * (9 / 31))

    return (valid_set_day_start, valid_set_day_end)


def load_bool_mask_from_file(mask_file_path):
    mask_np = np.load(mask_file_path)
    assert type(mask_np) == np.ndarray
    assert mask_np.dtype == np.dtype(
        'bool') or mask_np.dtype == np.dtype('float32')
    if mask_np.dtype == np.dtype('float32'):
        mask_bool = mask_np > 0
    else:
        mask_bool = mask_np
    return mask_bool


def load_torch_mask_from_file(mask_file_path):
    return torch.from_numpy(load_bool_mask_from_file(mask_file_path).astype(np.float32))


def mask_from_file_dimensions(mask_file_path):
    mask_np = np.load(mask_file_path)
    return mask_np.shape


def load_datasets(master_mask_file_name, evaluation_mask_file_name, training_mask_file_name,
                  training_mask_60_percent_file_name,
                  imputed_anom_file_name, val_set_percent, batch_size=32,
                  validation_mask_file_name=None, validation_mask_60_percent_file_name=None,
                  cuda_device="cuda:0", window_days=None):

    assert window_days is not None
    assert window_days > 0

    # Loading all masks and anomaly data in pytorch
    (duration, height, width) = mask_from_file_dimensions(
        converted_data_dir / evaluation_mask_file_name)

    # Loading anomaly and mask data

    evaluation_mask_3D_bool = load_bool_mask_from_file(
        converted_data_dir / evaluation_mask_file_name)
    evaluation_mask_3D = torch.from_numpy(
        evaluation_mask_3D_bool.astype(np.float32))

    training_mask_3D_bool = load_bool_mask_from_file(
        converted_data_dir / training_mask_file_name)
    training_mask_3D = torch.from_numpy(
        training_mask_3D_bool.astype(np.float32))

    training_mask_60_percent_3D_bool = load_bool_mask_from_file(
        converted_data_dir / training_mask_60_percent_file_name)
    training_mask_60_percent_3D = torch.from_numpy(
        training_mask_60_percent_3D_bool.astype(np.float32))

    if validation_mask_file_name == None:
        validation_mask_3D_bool = evaluation_mask_3D_bool & (
            ~training_mask_3D_bool)
        validation_mask_3D = torch.from_numpy(
            validation_mask_3D_bool.astype(np.float32))
    else:
        validation_mask_3D = load_torch_mask_from_file(
            converted_data_dir / validation_mask_file_name)

    if validation_mask_60_percent_file_name == None:
        validation_mask_60_percent_3D_bool = evaluation_mask_3D_bool & (
            ~training_mask_60_percent_3D_bool)
        validation_mask_60_percent_3D = torch.from_numpy(
            validation_mask_60_percent_3D_bool.astype(np.float32))
    else:
        validation_mask_60_percent_3D = load_torch_mask_from_file(
            converted_data_dir / validation_mask_60_percent_file_name)

    global master_mask_2D
    master_mask_2D = load_torch_mask_from_file(
        converted_data_dir / master_mask_file_name)

    anom_imputed_3D = torch.from_numpy(
        np.load(converted_data_dir / imputed_anom_file_name)).div_(4)  # divided by 4!

    anom_training_3D = anom_imputed_3D * training_mask_3D
    anom_training_60_percent_3D = anom_imputed_3D * training_mask_60_percent_3D

    day_idx = torch.arange(len(anom_imputed_3D), dtype=torch.float32)

    # Creation of dataset with all data
    all_ds = MyNDayDataset(window_days,
                           anom_training_3D,
                           training_mask_3D,
                           day_idx,
                           anom_imputed_3D,
                           evaluation_mask_3D)

    all_ds_60p = MyNDayDataset(window_days,
                               anom_training_60_percent_3D,
                               training_mask_60_percent_3D,
                               day_idx,
                               anom_imputed_3D,
                               evaluation_mask_3D)

    # Creation of main training and validation dataset
    (valid_set_day_start, valid_set_day_end) = get_validation_set_position(
        duration, val_set_percent)

    train_day_idxs = list(range(-all_ds.window_start, valid_set_day_start)) + \
        list(range(valid_set_day_end, len(day_idx) - all_ds.window_end))
    train_ds = Subset(all_ds, train_day_idxs)

    valid_day_idxs = list(range(valid_set_day_start, valid_set_day_end))
    valid_ds = Subset(all_ds, valid_day_idxs)
    valid_ds_60p = Subset(all_ds_60p, valid_day_idxs)

    data = DataBunch.create(train_ds=train_ds, valid_ds=valid_ds,
                            device=torch.device(cuda_device), bs=batch_size)
    data.add_tfm(add_noise_in_holes)

    # dataset only used for validation on 60 percent mask
    data_60p = DataBunch.create(train_ds=train_ds, valid_ds=valid_ds_60p, device=torch.device(cuda_device),
                                bs=batch_size)
    data_60p.add_tfm(add_noise_in_holes)

    # Creation of additional datasets used for different norms calculation
    valid_ds_all = Subset(
        all_ds, range(-all_ds.window_start, len(day_idx) - all_ds.window_end))
    data_all = DataBunch.create(train_ds=train_ds, valid_ds=valid_ds_all, device=torch.device(cuda_device),
                                bs=batch_size)
    data_all.add_tfm(add_noise_in_holes)

    valid_ds_DV = MyNDayDataset(window_days,
                                anom_training_3D,
                                training_mask_3D,
                                day_idx,
                                anom_imputed_3D,
                                validation_mask_3D)
    valid_ds_DV_60p = MyNDayDataset(window_days,
                                    anom_training_60_percent_3D,
                                    training_mask_60_percent_3D,
                                    day_idx,
                                    anom_imputed_3D,
                                    validation_mask_60_percent_3D)

    valid_ds_DV_window = Subset(
        valid_ds_DV, range(-all_ds.window_start, len(day_idx) - all_ds.window_end))
    data_DV = DataBunch.create(train_ds=train_ds, valid_ds=valid_ds_DV_window,
                               device=torch.device(cuda_device), bs=batch_size)
    data_DV.add_tfm(add_noise_in_holes)

    valid_ds_DV_on_v = Subset(valid_ds_DV, valid_day_idxs)
    data_DV_on_v = DataBunch.create(
        train_ds=train_ds, valid_ds=valid_ds_DV_on_v, device=torch.device(cuda_device), bs=batch_size)
    data_DV_on_v.add_tfm(add_noise_in_holes)

    # dataset only used for validation on 60 percent mask
    valid_ds_DV_on_v_60p = Subset(valid_ds_DV_60p, valid_day_idxs)
    data_DV_on_v_60p = DataBunch.create(train_ds=train_ds, valid_ds=valid_ds_DV_on_v_60p,
                                        device=torch.device(cuda_device),
                                        bs=batch_size)
    data_DV_on_v_60p.add_tfm(add_noise_in_holes)

    valid_ds_ident = MyNDayDataset(window_days,
                                   anom_imputed_3D,
                                   evaluation_mask_3D,
                                   day_idx,
                                   anom_imputed_3D,
                                   evaluation_mask_3D)
    valid_ds_ident = Subset(
        valid_ds_ident, range(-all_ds.window_start, len(day_idx) - all_ds.window_end))
    data_ident = DataBunch.create(
        train_ds=train_ds, valid_ds=valid_ds_ident, device=torch.device(cuda_device), bs=batch_size)
    data_ident.add_tfm(add_noise_in_holes)

    return (data, data_all, data_DV, data_DV_on_v, data_ident, data_60p, data_DV_on_v_60p, duration, height, width)
