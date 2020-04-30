#!/usr/bin/env python
# coding: utf-8

# Model

# Downsample verzija

# Bazira se na train_ensemble_batch_ds_3.py. Uzima hiperparametare za treniranje modela iz argumenata komandne linije.


# Load libraries

from fastai.imports import *
from fastai.basics import *
from fastai.callbacks import *

from ranger import ranger  # For new optimizer

import numpy as np
import pandas as pd
import glob
import parse

import pathlib

import os  # for deleting temp files
import time  # time measuring for logging purpose

import sys
import getopt

from distutils.util import strtobool

# Load data
from load_data_batch_3 import *

from model import masked_ABS, masked_MSE, ConvEncDec

from flat_cosine_mod import fit_fc_mod


# Function to check if a model allready exists on disk
def model_exist(model_num, model_name):
    model_dir = pathlib.Path("models")
    current_model_name = model_name + ".num=" + str(model_num)
    current_model_name_path = model_dir / (current_model_name + ".pth")
    if current_model_name_path.is_file():
        return True  # This model is allready calculated
    return False  # This model does not exist


def train_and_validate_model(model_num, window_days, encode_position, height, width, dim, inp_lay, red_lay, out_lay, red_exp, kernel_size,
                             max_lr, wd, epohs, dropout, loss_func, cuda_device, model_name,
                             masked_ratio, masked_ratio_prec, val_set_percent,
                             data, data_all, data_DV, data_DV_on_v, data_ident, data_60p, data_DV_on_v_60p,
                             batch_size, training_mask_file_name_suffix):

    # Auxiliary function for output formating
    def format_losses(losses):
        str_out = ""
        for loss in losses:
            str_out += str(loss.numpy())+" "

        return str_out

    def log_pre_training(file_name, file_name2, model, model_num, window_days, encode_position, dim, red_lay, red_exp, inp_lay, out_lay, latent_size,
                         max_lr, epohs, dropout, batch_size, loss_func, training_mask_file_name_suffix):

        # Generating log file model info
        output = str(model_num) + ". window_days = " + str(window_days) + ", encode_position = " + str(encode_position) + \
            ", dim = " + str(dim) + ", red_lay = " + str(red_lay) + ", red_exp = " + str(red_exp) + \
            ", inp_lay = " + str(inp_lay) + ", out_lay = " + str(out_lay) + ", latent_size = " + str(latent_size) + \
            ", Num. param = " + str(model.count_parameters()) + ", max_lr = " + str(max_lr) + \
            ", epohs = " + str(epohs) + ", dropout = " + \
            str(dropout) + ", batch_size = " + str(batch_size)

        if loss_func == masked_MSE:
            output += ", norm = MSE (L2)"
            loss_func_str = "MSE (L2)"
        elif loss_func == masked_ABS:
            loss_func_str = "ABS (L1)"
            output += ", norm = ABS (L1)"
        else:
            loss_func_str = "UNDEFINED"

        print(output)

        # Writing to log files
        file_out = open(file_name, "a")
        # log file gets "Loading datasets..." but screen output no; screen got it before ;-)
        file_out.write("Loading datasets with " +
                       training_mask_file_name_suffix + "\n" + output + "\n")
        file_out.close()

        output = "\n" + str(model_num) + ".\n"
        file_out2 = open(file_name2, "a")
        file_out2.write(output)
        file_out2.close()

        return loss_func_str

    def log_post_training(file_name2, epohs, max_lr, wd, num_train_losses, train_losses, valid_losses):

        file_out2 = open(file_name2, "a")
        file_out2.write("Epohs = "+str(epohs)+", max_lr = " +
                        str(max_lr)+", wd = "+str(wd)+"\n")
        file_out2.write("Training loss entries = "+str(num_train_losses)+"\n")
        file_out2.write("Training losses = " + train_losses + "\n")
        file_out2.write("Validation losses = " + valid_losses + "\n")
        file_out2.close()

    # Skipping already calculated models if model file_name allready exist
    if model_exist(model_num, model_name):
        return 1  # This model is allready calculated

    # Log files creation
    
    
    log_dir = pathlib.Path("logs")
    try:
        os.mkdir(log_dir)
    except FileExistsError:
        pass
    
    file_name = log_dir / (model_name + ".num=" + str(model_num) + "_log.txt")
    file_name2 = log_dir / (model_name + ".num=" +
                            str(model_num) + "_log-detailes.txt")

    if file_name.is_file():
        file_out = open(file_name, "a")
        file_out.write("\n\n...RESTARTING INTERRUPTED PROCESS...\n\n")
    else:
        file_out = open(file_name, "a")
        file_out.write("Final models generation\n----------------\n")
    file_out.close()

    if file_name2.is_file():
        file_out2 = open(file_name2, "a")
        file_out2.write("\n\n...RESTARTING INTERRUPTED PROCESS...\n\n")
    else:
        file_out2 = open(file_name2, "a")
        file_out2.write("Training and validation losses\n----------------\n")
    file_out2.close()

    # Model path and name variables
    model_dir = pathlib.Path("models")
    current_model_name = model_name + ".num=" + str(model_num)

    # Model object creation
    if cuda_device == "cpu" :
        model = ConvEncDec(height, width, window_days=window_days, out_channels=1, dim=dim, enc_dropout=dropout,
                           input_layers=inp_lay, reducing_layers=red_lay, output_layers=out_lay,
                           reduction_exponent=red_exp, kernel_size=kernel_size, encode_position=encode_position).cpu()
    else :        
        model = ConvEncDec(height, width, window_days=window_days, out_channels=1, dim=dim, enc_dropout=dropout,
                           input_layers=inp_lay, reducing_layers=red_lay, output_layers=out_lay,
                           reduction_exponent=red_exp, kernel_size=kernel_size, encode_position=encode_position).cuda()

    # Size of latent vector in model
    latent_size = np.prod(model.get_latent_vector_size())

    # Pre-training logging of model and optimizer hyperparameters
    loss_func_str = log_pre_training(file_name, file_name2, model, model_num, window_days, encode_position,
                                     dim, red_lay, red_exp, inp_lay,
                                     out_lay, latent_size, max_lr, epohs, dropout, batch_size, loss_func,
                                     training_mask_file_name_suffix)

    # Learner object creation (part of fastai library)
    #learner = Learner(data, model, loss_func=loss_func, opt_func=AdamW)
    learner = Learner(data, model, loss_func=loss_func, opt_func=ranger.Ranger)

    # Deleting temp model files
    for fn in [model_dir / (current_model_name + "-bestmodel.pth")]:
        if fn.is_file():
            os.remove(fn)

    # Time measuring
    time_total_beg = time.perf_counter()
    time_CPU_beg = time.process_time()

    # Finding optimal learning rate
    # learner.lr_find()
    # learner.recorder.plot()

    # Logging training and validation losses from learning rate finding
    #num_lr_train_losses = len(learner.recorder.losses)
    #lr_train_losses = format_losses(learner.recorder.losses)
    #file_out2 = open(file_name2, "a")
    #file_out2.write("Learning rate entries = "+str(num_lr_train_losses)+"\n")
    #file_out2.write("Learning rate losses = " + lr_train_losses + "\n")
    # file_out2.close()
    num_lr_train_losses = None
    lr_train_losses = None

    # Main model optimizer function call (fastai library)

    # Pre flat cosine training
    # learner.fit_one_cycle(epohs, max_lr=max_lr, wd=wd,
    #                      callbacks=[SaveModelCallback(learner, name=model_name+"-bestmodel")])

    # Before Tivek mod
    #learner.fit_fc(tot_epochs = epohs, wd=wd, lr=max_lr, callbacks=[SaveModelCallback(learner, name=current_model_name+"-bestmodel")])

    # After Tivek mod
    fit_fc_mod(learner, epohs, max_lr=max_lr, wd=wd, callbacks=[
               SaveModelCallback(learner, name=current_model_name+"-bestmodel")])

    # Logging training and validation losses after model training has finished
    num_train_losses = len(learner.recorder.losses)
    train_losses = format_losses(learner.recorder.losses)
    valid_losses = str(learner.recorder.val_losses).strip('[]')

    # Post-training logging of optimizer hyperparameters and losses
    log_post_training(file_name2, epohs, max_lr, wd,
                      num_train_losses, train_losses, valid_losses)

    # Calculating different validation losses for model benchmarking
    print("Computing val...")
    val = (learner.validate())[0]

    for loss_func_iter in [masked_ABS, masked_MSE]:
        output = ""

        norm_str = "UNKNOWN norm!!!"
        if loss_func_iter == masked_ABS:
            norm_str = "ABS norm"
        if loss_func_iter == masked_MSE:
            norm_str = "MSE norm"
        output += "Validation losses for "+norm_str+":\n"

        learner_60p = Learner(
            data_60p, model, loss_func=loss_func_iter, opt_func=AdamW)
        print(norm_str+": Computing val_60p...")
        val_60p = (learner_60p.validate())[0]
        del learner_60p

        learner_DV = Learner(
            data_DV, model, loss_func=loss_func_iter, opt_func=AdamW)
        print(norm_str+": Computing val_DV...")
        val_DV = (learner_DV.validate())[0]
        del learner_DV

        learner_all = Learner(
            data_all, model, loss_func=loss_func_iter, opt_func=AdamW)
        print(norm_str+": Computing val_all...")
        val_all = (learner_all.validate())[0]
        del learner_all

        learner_DV_on_v = Learner(
            data_DV_on_v, model, loss_func=loss_func_iter, opt_func=AdamW)
        print(norm_str+": Computing val_DV_on_v...")
        val_DV_on_v = (learner_DV_on_v.validate())[0]
        del learner_DV_on_v

        learner_DV_on_v_60p = Learner(
            data_DV_on_v_60p, model, loss_func=loss_func_iter, opt_func=AdamW)
        print(norm_str+": Computing val_DV_on_v_60p...")
        val_DV_on_v_60p = (learner_DV_on_v_60p.validate())[0]
        del learner_DV_on_v_60p

        learner_ident = Learner(
            data_ident, model, loss_func=loss_func_iter, opt_func=AdamW)
        print(norm_str+": Computing val_ident...")
        val_ident = (learner_ident.validate())[0]
        del learner_ident

        output += "vl_DV = " + str(val_DV) + ", vl_all = " + str(val_all) + \
                  ", vl_DV_on_v = " + str(val_DV_on_v) + ", vl_ident = " + str(val_ident) +\
                  ", vl_60p = " + str(val_60p) + \
            ", vl_DV_on_v_60p = " + str(val_DV_on_v_60p)

        if loss_func_iter == loss_func:
            vl_tl_ratio = learner.recorder.val_losses[-1] / \
                (learner.recorder.losses[-1].numpy())
            vl_last = learner.recorder.val_losses[-1]
            vl_min = min(learner.recorder.val_losses)
            output += ", vl = " + str(val) + ", vl/tl ratio = " + str(round(vl_tl_ratio, 4)) + \
                      ", vl_last/vl_min ratio = " + str(round(vl_last/vl_min, 4)) + \
                      ", vl_last = " + str(vl_last) + \
                ", vl_min = " + str(vl_min)

        if loss_func_iter == masked_ABS:
            val_DV_ABS = val_DV
            val_all_ABS = val_all
            val_DV_on_v_ABS = val_DV_on_v
            val_ident_ABS = val_ident
            val_60p_ABS = val_60p
            val_DV_on_v_60p_ABS = val_DV_on_v_60p

        if loss_func_iter == masked_MSE:
            val_DV_MSE = val_DV
            val_all_MSE = val_all
            val_DV_on_v_MSE = val_DV_on_v
            val_ident_MSE = val_ident
            val_60p_MSE = val_60p
            val_DV_on_v_60p_MSE = val_DV_on_v_60p

        print(output)

        # Saving to log file
        file_out = open(file_name, "a")
        file_out.write(output+"\n")
        file_out.close()

    # Logging elapsed time
    time_total = round(time.perf_counter()-time_total_beg)
    time_CPU = round(time.process_time()-time_CPU_beg)
    output2 = "----> (time_total, time_CPU) = (" + \
        str(time_total)+"s, " + str(time_CPU)+"s)"
    print(output2)
    print(" ")

    file_out = open(file_name, "a")
    file_out.write(output2+"\n\n")
    file_out.close()

    # Initialising database
    from database_categories_spec import database_categories
    output_database = pd.DataFrame(columns=database_categories)

    # Writing to database
    datarow = pd.Series([model_num, masked_ratio, masked_ratio_prec, height, width, window_days, encode_position,
                         dim, red_lay, red_exp, inp_lay, out_lay,
                         kernel_size, latent_size, model.count_parameters(
                         ), max_lr, wd, epohs, dropout, loss_func_str,
                         batch_size,
                         val, vl_tl_ratio, vl_last, vl_min,
                         val_DV_ABS, val_DV_MSE, val_all_ABS, val_all_MSE, val_DV_on_v_ABS, val_DV_on_v_MSE,
                         val_ident_ABS, val_ident_MSE, val_60p_ABS, val_60p_MSE, val_DV_on_v_60p_ABS, val_DV_on_v_60p_MSE,
                         val_set_percent, time_total, time_CPU, model_name,
                         num_lr_train_losses, lr_train_losses, num_train_losses, train_losses, valid_losses],
                        index=database_categories)

    output_database = output_database.append(datarow, ignore_index=True)
    output_database.to_csv(model_dir / (current_model_name + ".csv"))

    # Saving current trained model
    learner.save(current_model_name)
    del learner

    # Deleting temp model files
    for fn in [model_dir / (current_model_name + "-bestmodel.pth")]:
        if fn.is_file():
            os.remove(fn)

    del model
    torch.cuda.empty_cache()

    return 0


def main(argv):
    # Defaults. Variables having values None must be passed through command line arguments.
    model_name = None
    model_num = None
    masked_ratio = None
    masked_ratio_prec = 0.005  # all generated masks are this precision
    val_set_percent = 5 / 31  # prescribed validation set size
    dropout = None
    max_lr = None
    wd = None
    epohs = None
    batch_size = None
    window_days = None
    encode_position = None
    dim = None
    inp_lay = None
    red_lay = None
    out_lay = None
    red_exp = 1.0
    kernel_size = 5
    loss_func = None
    CPU_only = False

    # CUDA device number used for GPU computation. In single GPU systems should be set to 0
    cuda_device_no = 0

    # Parsing command line arguments
    try:
        opts, args = getopt.getopt(argv, "h", ["help", "model_name=", "model_num=", "masked_ratio=", "masked_ratio_prec=", "val_set_percent=",
                                               "dropout=", "max_lr=", "wd=", "epohs=", "batch_size=", "window_days=", "encode_position=",
                                               "dim=", "inp_lay=", "red_lay=", "out_lay=", "red_exp=", "kernel_size=", "loss_func=",
                                               "cuda_device_no=", "CPU-only"])
    except getopt.GetoptError:
        print(sys.argv[0] + ' --model_name=Name --model_num=Num ...')
        print(sys.argv[0] + ' --help  for more options')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(sys.argv[0] + ' --model_name=Name --model_num=Num ...')
            print("Options:")
            print("  --model_name=Name   Name of model to train. Default is", model_name)
            print("  --model_num=Num     Number of model to train. Default is", model_num)
            print("More options available. See source code.")
            sys.exit()
        elif opt in ("--model_name"):
            try:
                model_name = arg
                print("Model name is set to", model_name)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--model_num"):
            try:
                model_num = int(arg)
                print("Model number is set to", model_num)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--masked_ratio"):
            try:
                masked_ratio = float(arg)
                print("Masked ratio is set to", masked_ratio)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--dropout"):
            try:
                dropout = float(arg)
                print("Dropout is set to", dropout)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--max_lr"):
            try:
                max_lr = float(arg)
                print("Max learning rate is set to", max_lr)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--wd"):
            try:
                wd = float(arg)
                print("Weight decay is set to", wd)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--epohs"):
            try:
                epohs = int(arg)
                print("Epohs is set to", epohs)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--batch_size"):
            try:
                batch_size = int(arg)
                print("Batch size is set to", batch_size)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--dim"):
            try:
                dim = int(arg)
                print("Number of channels (dim) is set to", dim)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--window_days"):
            try:
                window_days = int(arg)
                print("Length of window of days is set to", window_days)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--encode_position"):
            try:
                encode_position = bool(strtobool(arg))
                print("Positional encoding is set to", encode_position)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--inp_lay"):
            try:
                inp_lay = int(arg)
                print("Input layers is set to", inp_lay)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--red_lay"):
            try:
                red_lay = int(arg)
                print("Reduction layers is set to", red_lay)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--out_lay"):
            try:
                out_lay = int(arg)
                print("Output layers is set to", out_lay)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--loss_func"):
            try:
                if arg in ["L1", "ABS"]:
                    loss_func = masked_ABS
                    print("Loss function is set to ABS (L1)")
                elif arg in ["L2", "MSE"]:
                    loss_func = masked_MSE
                    print("Loss function is set to MSE (L2)")
                else:
                    raise ValueError
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--cuda_device_no"):
            try:
                cuda_device_no = int(arg)
                print("CUDA device number set to", cuda_device_no)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("--CPU-only"):
            CPU_only = True
            print("Computation set to CPU only mode. Parameter CUDA device number is unused.")
        else:
            print("Default argument change not supported:", opt, arg)
            sys.exit("Bad argument.")

    # Check that all hyperparameters are set
    assert (model_name is not None) and (model_num is not None)
    assert (masked_ratio is not None) and (dropout is not None) and (max_lr is not None) and (
        wd is not None) and (epohs is not None) and (batch_size is not None)
    assert (window_days is not None) and (window_days > 0)
    assert (encode_position is not None)
    assert (dim is not None) and (inp_lay is not None) and (
        red_lay is not None) and (out_lay is not None) and (loss_func is not None)

    # Global setting for used CUDA device
    if CPU_only :
        cuda_device = "cpu"
    else :      
        torch.cuda.set_device(cuda_device_no)
        cuda_device = f"cuda:{cuda_device_no:d}"

    # If model already exist from before, terminate program
    if model_exist(model_num, model_name):
        print("...skipping current model num = " +
              str(model_num) + ". File allready exists!")
        return 1

    # Setting mask file name. Using ds_3x3 masks is hardcoded
    base_training_mask_file_name = "training_mask_3D_shifted-v2"
    postfix_training_mask_file_name = "_ds_3x3.npy"
    training_mask_file_name_suffix = base_training_mask_file_name + "-" + str(masked_ratio) + "-" + \
        str(masked_ratio_prec) + postfix_training_mask_file_name

    # Loading downsampled datasets. Hardcoded use of ds_3x3 data
    print("Loading datasets with", training_mask_file_name_suffix)
    (data, data_all, data_DV, data_DV_on_v, data_ident, data_60p, data_DV_on_v_60p, duration, height, width) = load_datasets(
        master_mask_file_name="master_mask_2D_shifted-v2_ds_3x3.npy",
        evaluation_mask_file_name="evaluation_mask_3D_shifted-v2_ds_3x3.npy",
        training_mask_file_name=training_mask_file_name_suffix,
        training_mask_60_percent_file_name="training_mask_3D_shifted-v2-0.6-0.005_ds_3x3.npy",
        imputed_anom_file_name="imputed_anom_3D_shifted-v2_ds_3x3.npy",
        val_set_percent=val_set_percent,
        batch_size=batch_size,
        validation_mask_file_name="validation_mask_3D_shifted-v2_ds_3x3_from_training_mask_3D_shifted-v2-" +
        str(masked_ratio) + "-" +
        str(masked_ratio_prec) + ".npy",
        validation_mask_60_percent_file_name="validation_mask_3D_shifted-v2_ds_3x3_from_training_mask_3D_shifted-v2-0.6-0.005.npy",
        cuda_device=cuda_device,
        window_days=window_days)

    print("Model training and validation...")
    assert train_and_validate_model(model_num, window_days, encode_position, height, width, dim, inp_lay, red_lay, out_lay, red_exp, kernel_size,
                                    max_lr, wd, epohs, dropout, loss_func, cuda_device, model_name,
                                    masked_ratio, masked_ratio_prec, val_set_percent,
                                    data, data_all, data_DV, data_DV_on_v, data_ident, data_60p, data_DV_on_v_60p,
                                    batch_size, training_mask_file_name_suffix) == 0

    print("-----")
    print(" ")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
