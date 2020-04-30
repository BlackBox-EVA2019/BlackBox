#!/usr/bin/env python
# coding: utf-8

# Final script for generating predictions from the ensemble of trained models.
# Reads trained model information from .csv file and reads model from models directory.
# Automaticly works for all sizes of models (downsampled or full size models).


# Load libraries

import pathlib
import glob
import os  # for deleting temp files
import time  # time measuring for logging purpose
import sys
import getopt
import math  # isnan

from fastai.imports import *
from fastai.basics import *
from fastai.callbacks import *

from tqdm import tqdm  # for progress bars

import numpy as np
import pandas as pd
import torch

from statsmodels.distributions.empirical_distribution import ECDF

# R related input and output handling
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
numpy2ri.activate()


# load data
from load_data_batch_3 import *

# load twCRPS
from evaluate_score import twCRPS

# load model definition and loss functions
from model import masked_ABS, masked_MSE, ConvEncDec


# Loading distances data
dist_bool = np.load(converted_data_dir / "dist.npy") <= 50  # Radius is 50 km


# Definition of functions used for generating minimum prediction distributions on validation set
datashape = (11315, 16703)


# Global variables with masks and anomaly base file name
master_mask_base_file_name = "master_mask_2D_shifted-v2"
evaluation_mask_base_file_name = "evaluation_mask_3D_shifted-v2"
training_mask_60_percent_base_file_name = "training_mask_3D_shifted-v2-0.6-0.005"
imputed_anom_base_file_name = "imputed_anom_3D_shifted-v2"

# Functions used to make masks and anomaly file names
def get_name_suffix(factor) :
    if factor > 1 :
        return "_ds_" + str(factor) + "x" + str(factor)
    else :
        return ""

def get_file_name(base_file_name, factor=1) :
    return base_file_name + get_name_suffix(factor) + ".npy"


# Computation of cylindrical neighborhoods
def neighborhood(day, loc):
    timestart = max(0, day - 3)
    timeend = min(day + 4, datashape[0])
    return np.ix_(range(timestart, timeend), dist_bool[loc])


# Computation of minimums on cylindrical
def extremes_on_indices(anomaly, idx):
    return np.fromiter((anomaly[neighborhood(day, loc)].min() for day, loc in zip(*idx)), dtype=np.float64)


# Computation of discrete distributions
def prediction_from_extremes(Xv):
    assert Xv.shape[0] == 162000
    assert len(Xv.shape) == 2
    
    xk = -1 + np.arange(1.0, 401.0) / 100

    def sample_ecdf(X):
        return ECDF(X)(xk)
    
    return np.apply_along_axis(sample_ecdf, 1, Xv)


# Upsampling prediction from downsampled model. Using special hack that helps better interpolation arround the boundary.
def upsample_with_boundary_pixels_hack(prediction_ds, mask_2D_ds_t, mask_2D_us_bool) :
    # mask_2D_ds_t is tensor fractional downsampled mask
    
    mask_2D_ds = mask_2D_ds_t.numpy()
    assert len(prediction_ds.shape) == 3
    assert len(mask_2D_ds.shape) == 2
    assert len(mask_2D_us_bool.shape) == 2

    (height, width) = mask_2D_us_bool.shape
    (new_height, new_width) = mask_2D_ds.shape
    duration = prediction_ds.shape[0]
    assert new_height == prediction_ds.shape[1]
    assert new_width == prediction_ds.shape[2]
    
    assert height % new_height == 0
    assert width % new_width == 0
    factor = height // new_height
    assert factor == width // new_width
    
    prediction_ds_T = torch.from_numpy(prediction_ds).unsqueeze(1)
    
    anom_avg = torch.nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=False,
                                  count_include_pad=False)(prediction_ds_T)
    
    big_mask = torch.mul(mask_2D_ds_t, factor*factor)
    
    zeros = torch.zeros(duration, 1, new_height, new_width, dtype=torch.float32)
    ones = torch.ones(duration, 1, new_height, new_width, dtype=torch.float32)
    
    true_mask = torch.where(big_mask > 0, ones, zeros)
    del ones
    
    avg_mask = torch.nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=False,
                                  count_include_pad=False)(true_mask)
    del true_mask
    
    anom_avg_boundary = torch.where(big_mask > 0, zeros, anom_avg)
    del big_mask, anom_avg
    
    anom_true_boundary = torch.div(anom_avg_boundary, avg_mask)
    del anom_avg_boundary, avg_mask
    
    nan_mask = torch.isnan(anom_true_boundary)
    
    anom_corr_boundary = torch.where(nan_mask, zeros, anom_true_boundary)
    del nan_mask, zeros, anom_true_boundary
    
    anom_corr = torch.add(prediction_ds_T, anom_corr_boundary)
    del prediction_ds_T, anom_corr_boundary
    
    prediction_us_corr_T = torch.nn.functional.interpolate(
        anom_corr.cuda(), scale_factor = factor, mode = 'bicubic', align_corners=False).cpu()
    del anom_corr
    
    return prediction_us_corr_T.squeeze().numpy() * mask_2D_us_bool


# Generating final data for scoring purpose
def generate_scoring_distribution(model, window_days, number_of_predictions=1, factor=1, cuda_device=None):
    
# All necessary data loading
    print("...Loading data...")
    
    # Take care that master_mask_2D is global variable defined elsewhere!
    
    # Model size master mask
    master_mask_2D_bool = load_bool_mask_from_file(converted_data_dir / get_file_name(master_mask_base_file_name, factor))
    
    # Possibly fractional master mask tensor of model size - for upsampling
    if factor > 1 :
        master_mask_2D_t = torch.from_numpy(np.load(converted_data_dir / get_file_name(master_mask_base_file_name, factor)))
    
    # Full size master mask
    if factor > 1 :
        master_mask_2D_fs_bool = load_bool_mask_from_file(converted_data_dir / get_file_name(master_mask_base_file_name))
    
    # Model size evaluation mask
    evaluation_mask_3D_bool = load_bool_mask_from_file(
        converted_data_dir / get_file_name(evaluation_mask_base_file_name, factor))
    (duration, height, width) = evaluation_mask_3D_bool.shape
    evaluation_mask_3D = torch.from_numpy(evaluation_mask_3D_bool.astype(np.float32))
    del evaluation_mask_3D_bool
    
    # Full size evaluation mask
    if factor > 1 :
        evaluation_mask_3D_fs = load_torch_mask_from_file(converted_data_dir / get_file_name(evaluation_mask_base_file_name))
    
    # Model size imputed anomaly
    anom_imputed_3D = torch.from_numpy(np.load(converted_data_dir /
                                               get_file_name(imputed_anom_base_file_name, factor))).div_(4) # divided by 4!
    # day integers...
    day_idx = torch.arange(len(anom_imputed_3D), dtype=torch.float32)

    # Full size imputed anomaly (NOT divided by 4!)
    if factor > 1 :
        anom_imputed_3D_fs = torch.from_numpy(np.load(converted_data_dir / get_file_name(imputed_anom_base_file_name)))
        
    # Full size data is created and will be used only when factor > 1 and only in the upsampling phase
    
    # Loading validation indexes data
    robjects.r.load('DATA_TRAINING.RData')
    index_validation = np.array(robjects.r["index.validation"]) - 1
    index_validation_2D = np.unravel_index(index_validation, datashape, order='F') # R arrays have FORTRAN-style raveling
    del index_validation  # reducing memory usage

    #window_days = model.window_days
    window_start = - (window_days // 2)
    window_end = window_days - (window_days // 2)

    assert(index_validation_2D[0].min() + window_start >= 0)
    assert(index_validation_2D[0].max() + window_end <= duration)

    eval_ds = MyNDayDataset(window_days,
                            anom_imputed_3D,
                            evaluation_mask_3D,
                            day_idx,
                            anom_imputed_3D,
                            evaluation_mask_3D)
    eval_ds = Subset(eval_ds, range(day_d, len(day_idx) - window_end))
    databunch = DataBunch.create(train_ds=eval_ds, valid_ds=eval_ds, device=torch.device(
        cuda_device), bs=64)  # , num_workers=4)
    databunch.add_tfm(add_noise_in_holes)

    # Precomputed row and shifted column indexing data
    row_idxs, col_idxs = [np.load(f) for f in (converted_data_dir / "row_idxs-v2.npy",
                                               converted_data_dir / "shifted_col_idxs-v2.npy")]
        
    
# Main loop over the number of predictions to generate
    Xv_samples = []
    with tqdm(range(number_of_predictions)) as master_pbar:
        for num_pred in master_pbar:
            master_pbar.set_description(
                f'Sample {num_pred + 1}/{number_of_predictions}')

            # Stores point predictions from model for whole history
            prediction_all = np.zeros((duration, height, width), dtype=np.float32)
            with torch.no_grad():
                # Model is evaluated on all available data
                # for day_shift, (X, mask) in tqdm(enumerate(zip(anom_imputed_3D[day_d:], evaluation_mask_3D[day_d:]))):
                for (Xs_t, masks_t, days_t), (X_t, mask_t) in tqdm(databunch.valid_dl, desc="Evaluating model", leave=False):
                    # Model evaluation for a single day
                    pred = model(Xs_t, masks_t, days_t)
                    # Final prediction is combination of known data (where available)
                    # and data predicted by model (where original data is missing)
                    pred_filled = (mask_t * X_t + (1 - mask_t)
                                   * pred) * 4  # Multiplied by 4!!!!!
                    targeted_days = days_t.cpu().numpy().astype(int)[
                        :, -window_start]
                    # Data copied back to RAM
                    prediction_all[targeted_days] = pred_filled.cpu().numpy()
            # Prediction is trimmed to master_mask shape
            prediction_all_trimmed = prediction_all * master_mask_2D_bool
            del prediction_all # reducing memory usage
            
            # Upsampling if factor > 1. Input and output prom this block are in prediction_all_trimmed
            if factor > 1 :
                master_pbar.write("......Upsampling model prediction...")
                
                prediction_all_us = upsample_with_boundary_pixels_hack(prediction_all_trimmed, master_mask_2D_t,
                                                                    master_mask_2D_fs_bool)
                del prediction_all_trimmed
                
                prediction_all_us_t = torch.from_numpy(prediction_all_us)
                del prediction_all_us
                
                # Again, upsampled final prediction is combination of known data (where available)
                # and upsampled data predicted by model (where original data is missing)
                prediction_all_trimmed_t = evaluation_mask_3D_fs * anom_imputed_3D_fs + \
                                            (1 - evaluation_mask_3D_fs) * prediction_all_us_t
                del prediction_all_us_t
                
                prediction_all_trimmed = prediction_all_trimmed_t.numpy()
                del prediction_all_trimmed_t
            
            # Converting to repaired data. 2D spatial matrix data is converted back to 1D original indexed format
            master_pbar.write("......Converting to repaired data...")
            #repaired = []
            # for day in range(len(prediction_all_trimmed)):
            #    flattened = prediction_all_trimmed[day, row_idxs, col_idxs]
            #    repaired.append(flattened)
            #repaired = np.stack(repaired)
            repaired = prediction_all_trimmed[:, row_idxs, col_idxs]
            del prediction_all_trimmed

            # Gathering sample predictions of minimums on the validation set
            # Xv_samples containes
            master_pbar.write("......Gathering sample distribution...")
            Xv_samples.append(extremes_on_indices(repaired, index_validation_2D))
            del repaired  # reducing memory usage

        
    # return minimum predictions for every point in validation set
    return np.stack(Xv_samples, axis=1)


# Main functionality - batch generating mock score predictions from database of trained models
def main(argv):
    input_database_name = None
    number_of_predictions = None
    prediction_suffix = None
    cuda_device_no = 0
    included_models_r_N_list = None
    included_models_r_N_list_str = None
    output_database_name = None
    multi_GPU_computing = False
    predictions_list = None
    local_ensemble = False
    global_ensemble = False
    
    # Parsing command line arguments
    try:
        opts, args = getopt.getopt(argv, "hi:c:n:o:S:LE", ["help", "input_database_name=", "cuda_device_no=", "include=",
                                                         "number_of_predictions=", "prediction_suffix=", "output_database_name=",
                                                          "multi_GPU_computing", "only_score=", "local_ensemble", "global_ensemble"])
    except getopt.GetoptError:
        print(sys.argv[0] + ' --help  for more options')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("USAGE: ", sys.argv[0] + ' --input_database_name=Name1 --output_database_name=Name2 --number_of_predictions=Num --prediction_suffix=Suffix [ADDITIONAL OPTIONS]...')
            print("Batch calculate scores for a family of models Name1, writing scores in Name2, sampling Num number of predictions for each model, naming predictions with Suffix.")
            print("  OR:  ", sys.argv[0] + ' --input_database_name=Name1 --output_database_name=Name2 -S N1:S1,N2:S2,... [-L] [-E] [ADDITIONAL OPTIONS]...')
            print("Batch calculate scores for a family of models Name1, writing scores in Name2, using an ensemble of precalculated N1,N2,... predictions for each model, with suffixes S1,S2,... (no GPU usage)")
            print("Optionally calculate score over all models (-E option) and/or score over first N models for N=1,2,... (-L option)")
            print("OPTIONS:")
            print("  -i Name,   --input_database_name=Name     Name of csv model database containing input trained models (without extension)")
            print("  -o Name,   --output_database_name=Name    Name of csv database containing score (without extension)")
            print("  --number_of_predictions=Num               Evaluate Num predictions for each model in input_model_database")
            print("  --prediction_suffix=Suffix                Prediction file name suffix, to use")
            print("  -S N1:S1,N2:S2,... --only_score=...       Only calculate score from precalculated ensemble of minima predictions (num_of_pred:suffix,...=N1:S1,...)")
            print("  -L         --local_ensemble               In case of calculating score from precalculated ensemble (-S), calculate score over first N models, for N=1,2,... (SLOW)")
            print("  -E,        --global_ensemble              In case of calculating score from precalculated ensemble (-S), calculate score over all models and write ensemble prediction in RData file")
            print("ADDITIONAL OPTIONS:")
            print("  -c No,     --cuda_device_no=No            CUDA device number (default:0)")
            print("  -n r1/N1,r2/N2,... --include=r1/N1,r2/N2  Only process models having numbers which give reminders r1, r2, ... divided by N1, N2, ...")
            print("  --multi_GPU_computing                     Use multiple GPUs for computing (available only for prediction creation)")
            sys.exit()
        elif opt in ("-i", "--input_database_name"):
            try:
                input_database_name = arg
                print("Input database name is set to", input_database_name)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("-c", "--cuda_device_no"):
            try:
                cuda_device_no = int(arg)
                print("CUDA device number set to", cuda_device_no)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("-n", "--include"):
            try:
                included_models_r_N_list_str = arg.replace("/","-").replace(",","_")
                included_models_r_N_list = [(int(l[0]),int(l[1])) for l in [token.split("/") for token in arg.split(",")]]
                print("Only training models [(r1,N1),(r2,N2),...]", included_models_r_N_list)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--number_of_predictions"):
            try:
                number_of_predictions = int(arg)
                print("Number of predictions for each model is set to", number_of_predictions)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--prediction_suffix"):
            try:
                prediction_suffix = arg
                print("Predictions suffix is set to", prediction_suffix)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("-o", "--output_database_name"):
            try:
                output_database_name = arg
                print("Output database name is set to", output_database_name)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--multi_GPU_computing"):
            try:
                multi_GPU_computing = True
                print("Using multi GPU computing")
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("-S", "--only_score"):
            try:
                predictions_list = [(int(l[0]),str(l[1])) for l in [token.split(":") for token in arg.split(",")]]
                print("Only calculate score from precalculated ensemble of minima predictions [(num_of_pred,suffix),...]", predictions_list)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("-L", "--local_ensemble"):
            try:
                local_ensemble = True
                print("Calculate score for all local ensembles")
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("-E", "--global_ensemble"):
            try:
                global_ensemble = True
                print("Calculate score for global ensemble")
            except ValueError:
                sys.exit("Bad argument.")

                
    assert input_database_name is not None
    assert output_database_name is not None
    assert (predictions_list is not None) or ((number_of_predictions is not None) and (prediction_suffix is not None) and (not global_ensemble) and (not local_ensemble))    
    
    # Global setting for used CUDA device
    torch.cuda.set_device(cuda_device_no)
    cuda_device = f"cuda:{cuda_device_no:d}"

    # Loading database with calculated models info
    input_database_name_path = pathlib.Path(input_database_name + ".csv")
    assert input_database_name_path.is_file()
    database = pd.read_csv(input_database_name + ".csv")

    # Prepare output database with scores
    if predictions_list is not None :
        cumulative_number_of_predictions = 0
        cumulative_prediction_suffix = ""
        for pred in predictions_list :
            cumulative_number_of_predictions = cumulative_number_of_predictions + pred[0]
            cumulative_prediction_suffix = cumulative_prediction_suffix + "_" + pred[1]
        output_database_name= output_database_name + "_" + str(cumulative_number_of_predictions) + cumulative_prediction_suffix
        number_of_predictions = cumulative_number_of_predictions
        prediction_suffix = cumulative_prediction_suffix[1:]
    else :
        output_database_name= output_database_name + "_" + str(number_of_predictions) + "_" + prediction_suffix
    
    if included_models_r_N_list is not None :
        output_database_name = output_database_name + "_" + included_models_r_N_list_str
    output_database_name_path = pathlib.Path(output_database_name + ".csv")
    
    output_columns = ["model_num", "model_name", "number_of_predictions", "prediction_suffix", "score"]
    if predictions_list is None :
        if output_database_name_path.is_file():
            print("Output database allready exists. Continuing interrupted process.")
            output_database = pd.read_csv(output_database_name_path)
        else:
            output_database = pd.DataFrame(columns=output_columns)
    else:
        output_database = pd.DataFrame(columns=output_columns)

    # Create output directory for minima, if not exists
    output_data_dir = pathlib.Path("minima")
    try:
        os.mkdir(output_data_dir)
    except FileExistsError:
        pass

    # Create output directory for predictions, if not exists
    prediction_dir = pathlib.Path('predictions')
    try:
        os.mkdir(prediction_dir)
    except FileExistsError:
        pass

    # Get full size from master mask
    (height_fs, width_fs) = np.load(converted_data_dir / get_file_name(master_mask_base_file_name)).shape
    
    # Main loop iterating all calculated models from database
    print("Starting main loop iteration over input models...")
    Xv_all = []
    global_number_of_predictions = 0
    for row in tqdm(database.iterrows()):
        data_row = row[1]

        # Assigning local variable names
        model_num, model_name = data_row.model_num, data_row.model_name
        height, width, dim, dropout = data_row.height, data_row.width, data_row.dim, data_row.dropout
        inp_lay, red_lay, out_lay, red_exp = data_row.inp_lay, data_row.red_lay, data_row.out_lay, data_row.red_exp
        kernel_size, loss_func = data_row.kernel_size, data_row.loss_func
        val_set_percent = data_row.val_set_percent
        
        # Block to skip some models
        if included_models_r_N_list != None :
            train_flag = False
            for (r, N) in included_models_r_N_list :
                if model_num % N == r :
                    train_flag = True
                    break
            if not train_flag :  # skip training this model
                print("Skipping current model - because include=", included_models_r_N_list)
                continue
        
        # Checking if precalculated model file exists
        model_dir = pathlib.Path("models")
        current_model_name = model_name + ".num=" + str(model_num)
        current_model_name_path = model_dir / (current_model_name + ".pth")
        print("Model =", current_model_name)

        # Normal operation - minima prediction is calculated from trained model
        if predictions_list is None :
            # Assigning local variable names for names not present in older .csv version
            # To insure compatibility of alternative operation of just score calculation with older .csv version
            # These names are not used in score calculation from precalculated ensemble of minima predictions
            window_days = data_row.window_days
            encode_position = data_row.encode_position

            # Checking if input database (and so also models) are not from older incompatible type
            assert not math.isnan(window_days)
            assert not math.isnan(encode_position)

            # Calculating model reduction factor
            assert height_fs % height == 0 and width_fs % width == 0
            factor = height_fs // height
            assert factor == width_fs // width

            # Checking if predictions for particular model already exist
            samples_out_name = current_model_name + "_predictions_for_"+str(number_of_predictions)+"_samples_" + \
                                prediction_suffix + ".npz"
            samples_out_path = output_data_dir / samples_out_name
            if samples_out_path.is_file():
                print("...Prediction for current model already exist. Skipping to the next model...\n")
                continue  #  If predictions for particular model already exist, skip to the next model
            else :
                # Time measuring
                time_total_beg = time.perf_counter()
                time_CPU_beg = time.process_time()            

            # Writing log with current model information
            output = "h*w = (" + str(height) + ", " + str(width) + "), window_days = " + str(window_days) + \
                     ", encode_position = " + str(encode_position) + ", dim = " + str(dim) + \
                     ", red_lay = " + str(red_lay) + ", red_exp = " + str(red_exp) + ", inp_lay = " + str(inp_lay) + \
                     ", out_lay = " + str(out_lay) + ", latent_size = " + str(data_row.latent_size) + \
                     ", Num. param = " + str(data_row.num_param) + ", max_lr = " + str(data_row.max_lr) + \
                     ", epohs = " + str(data_row.epohs) + ", dropout = " + str(dropout)
            if loss_func == "MSE (L2)":
                output += ", norm = MSE (L2)"
            if loss_func == "ABS (L1)":
                output += ", norm = ABS (L1)"
            print(output)

            print("Loading model data from disk...")
            # Model object creation
            model = ConvEncDec(height, width, window_days=window_days, encode_position=encode_position,
                               out_channels=1, dim=dim, enc_dropout=dropout,
                               input_layers=inp_lay, reducing_layers=red_lay, output_layers=out_lay,
                               reduction_exponent=red_exp, kernel_size=kernel_size).cuda()

            # Creating data object for Learner
            val_set_percent = 5 / 31
            *_, height2, width2 = \
                load_datasets(get_file_name(master_mask_base_file_name, factor),
                              get_file_name(evaluation_mask_base_file_name, factor),
                              get_file_name(training_mask_60_percent_base_file_name, factor),
                              get_file_name(training_mask_60_percent_base_file_name, factor),
                              get_file_name(imputed_anom_base_file_name, factor),
                              val_set_percent = val_set_percent,
                              cuda_device = cuda_device,
                              window_days = window_days)
            del _

            # Height and width from database and from data masks should match
            assert height == height2 and width == width2

            # Loading trained model from file. By default it assumes models are saved in "models/" subdirectory
            assert current_model_name_path.is_file()
            model.load_state_dict(torch.load(current_model_name_path, map_location=cuda_device)["model"])
            if multi_GPU_computing :
                model=torch.nn.DataParallel(model)
            model.eval()

            # Main function call for generating distributions
            print("Generate minimum predictions...")
            Xv = generate_scoring_distribution(model, window_days, number_of_predictions, factor, cuda_device=cuda_device)

            # Force free memory
            del model
        
        # Alternative operation - only calculate score from precalculated ensemble of minima predictions
        else :
            print("Loading ensemble minimum predictions data...")
            Xv = []
            for pred in predictions_list :
                # Checking if precalculated minima predictions exist
                samples_out_name = current_model_name + "_predictions_for_"+str(pred[0])+"_samples_" + \
                                    pred[1] + ".npz"
                samples_out_path = output_data_dir / samples_out_name
                if samples_out_path.is_file():
                    sample = np.load(samples_out_path)
                    sample = sample[sample.files[0]]
                    Xv.append(sample)
                    if global_ensemble or local_ensemble : # to reduce large memory waste, if not using this options
                        Xv_all.append(sample)
                        global_number_of_predictions = global_number_of_predictions + pred[0]
                else :
                    sys.exit("...ERROR: Minima prediction " + samples_out_name + " does not exist!")
            Xv = np.concatenate(Xv,axis=1)
        
        # Generating distribution prediction
        print("Generating distribution prediction...")
        prediction = prediction_from_extremes(Xv)

        # Generate and save score
        print("Calculating score from distribution prediction...")
        score = twCRPS(prediction)
        print("Writing score to output database...")
        output_database = output_database.append(pd.Series([model_num,
                                                            model_name,
                                                            number_of_predictions,
                                                            prediction_suffix,
                                                            score],
                                                           index=output_columns),
                                                 ignore_index=True)
        output_database.to_csv(output_database_name_path)
        
        # Normal operation - minima prediction is saved
        if predictions_list is None :
            # Saving minimum predictions data for later postprocessing
            print("Saving minimum predictions data...")
            np.savez_compressed(samples_out_path, Xv)
            
            # Logging elapsed time
            time_total = round(time.perf_counter()-time_total_beg)
            time_CPU = round(time.process_time()-time_CPU_beg)
            print("Elapsed time ----> (time_total, time_CPU) = (" + str(time_total) + "s, " + str(time_CPU) + "s)\n")
        del prediction
        del Xv
        
        # Alternate operation - calculation local ensemble score
        if (predictions_list is not None) and local_ensemble :
            print("Preprocessing all current predictions...")
            Xv_all_concat = np.concatenate(Xv_all,axis=1)
            print("Generating local ensemble distribution prediction... (total current number of predictions =", (Xv_all_concat.shape)[1], ")")
            prediction = prediction_from_extremes(Xv_all_concat)
            del Xv_all_concat
            print("Calculating local ensemble score from distribution prediction...")
            score = twCRPS(prediction)
            del prediction
            print("Local ensemble score is", score)
            print("Writing local ensemble score to output database...")
            output_database = output_database.append(pd.Series([model_num,
                                                                model_name,
                                                                global_number_of_predictions,
                                                                prediction_suffix + "_FIRST_MODELS",
                                                                score],
                                                               index=output_columns),
                                                     ignore_index=True)
            output_database.to_csv(output_database_name_path)
        
        # Print iterations separator
        print(" ")


    

    # Alternate operation - calculation global ensemble score
    if (predictions_list is not None) and global_ensemble :
        Xv_all = np.concatenate(Xv_all,axis=1)
        print("Generating global ensemble distribution prediction... (total number of predictions =", (Xv_all.shape)[1], ")")
        prediction = prediction_from_extremes(Xv_all)
        print("Calculating global ensemble score from distribution prediction...")
        score = twCRPS(prediction)
        print("Global ensemble score is", score)
        print("Writing global ensemble score to output database...")
        output_database = output_database.append(pd.Series([0,
                                                            model_name,
                                                            global_number_of_predictions,
                                                            prediction_suffix + "_ALL_MODELS",
                                                            score],
                                                           index=output_columns),
                                                 ignore_index=True)
        output_database.to_csv(output_database_name_path)
            
        # Saving distribution prediction
        print("Saving global ensemble distribution prediction to RData...")
        prediction_out_name = output_database_name + "_global_ensemble_distribution_prediction_for_" + str(global_number_of_predictions) + \
                            "_samples_" + prediction_suffix + "_ALL_MODELS.RData"
        prediction_out_path = prediction_dir / prediction_out_name
        robjects.r.assign("prediction", prediction)
        robjects.r(f"save(prediction, file='{str(prediction_out_path)}')")
            

        
        
if __name__ == "__main__":
   main(sys.argv[1:])
