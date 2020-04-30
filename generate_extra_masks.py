#!/usr/bin/env python

# Generate new holes, big holes for validation of all models

# Generate additional random holes over (percent_masked) of remaining data of the whole history.
# Additional Holes are covering approximately half of existing data.
# Training mask is evaluation mask with additional new random holes removed.
# Training masks in "input_data/training_mask_3D-(percent_masked).npy"

import numpy as np
import random
import time
import pathlib
import sys
import getopt


def generate_random_mask_2D(num_points, master_mask_2D):
    """Generate random mask for a single day, having num_points.
    The code is hand-tweaked in such a way that this mask resembles the "look" of the original nan_mask."""

    (height, width) = master_mask_2D.shape
    random_nan_mask_2D = np.zeros((height, width), dtype=bool)
    num=0
    epsp=0.00008
    while(num<num_points):
        row=random.randint(0,height-1)
        col=random.randint(0,width-1)
        if(~random_nan_mask_2D[row,col]):
            s1 = random_nan_mask_2D[row-1:row+2,col-1:col+2].sum()
            s2 = random_nan_mask_2D[row-2:row+3,col-2:col+3].sum()
            if(s1>0):
                p=s1/8
            elif(s2>0):
                p=(s2/16)*0.02
            else:
                p=epsp

            if(p>random.random()):
                random_nan_mask_2D[row,col]=True
                if(master_mask_2D[row,col]):
                    num+=1
    
    return random_nan_mask_2D & master_mask_2D


def generate_random_mask_history(data_dir, percent_masked = 0.5, eps_p = 0.01):
    """Generate new random masks over the whole history. Masks are unique for every month.
    Parameters:
        percent_masked is the desired ration of masked data in remaining data (evaluation_mask_3D)
        eps_p is the desired absolute precision of mask generation
    Returns: training_mask_3D
    """
    
    # Data files input
    print("Loading data files...")
    master_mask_2D = np.load(data_dir / "master_mask_2D-v2.npy")
    nan_mask_3D = np.load(data_dir / "nan_mask_3D-v2.npy")
    evaluation_mask_3D = np.load(data_dir / "evaluation_mask_3D-v2.npy")
    (duration, height, width) = evaluation_mask_3D.shape
    
    print("Generating random mask...")
    print("...ratio of masked data is set to", percent_masked)
    print("...precision of generated masked data ratio is set to", eps_p)
    
    
    total_count_master_mask = master_mask_2D.sum()
    count_masked = round(total_count_master_mask * percent_masked)
    
    good_count = 0
    miss_count = 0
    max_miss_iter = 0
    
    random_mask_3D = np.zeros((duration, height, width), dtype=bool)
    t = 0
    time_zero = time.time()
    while t < duration:
        clear_flag = True

        print("Generating random mask for day", t)
        random_mask = generate_random_mask_2D(count_masked, master_mask_2D)
        union_mask = nan_mask_3D[t] | random_mask
        total_count = union_mask.sum()
        total_count_percentage = total_count / total_count_master_mask

        # Absolute random mask percentage depends upon the existing non_mask_3D size
        target_p = (nan_mask_3D[t].sum() / total_count_master_mask +
                    (1 - nan_mask_3D[t].sum() / total_count_master_mask) * percent_masked)

        # Generating random mask within given tolerance eps_p
        MAX_RAND_ITER = 1000
        for i in range(MAX_RAND_ITER):
            if (total_count_percentage >= target_p - eps_p) & (total_count_percentage <= target_p + eps_p) :
                print("...mask area (good) =", round(total_count_percentage*100, 2), "%")
                good_count += 1
                if i > max_miss_iter:
                    max_miss_iter = i
                    print("...current maximum number of miss iterations increased. Current max =", max_miss_iter)
                break
            else :
                miss_count += 1
                if(good_count != 0):
                    print("...mask area (bad) =", round(total_count_percentage*100, 2), "%, miss ratio =",
                          round(miss_count / good_count, 4), " ...regenerating random mask!")
                else:
                    print("...mask area (bad) =", round(total_count_percentage*100, 2), "% ...regenerating random mask!")                    
                random_mask = generate_random_mask_2D(count_masked, master_mask_2D)
                union_mask = nan_mask_3D[t] | random_mask
                total_count = union_mask.sum()
                total_count_percentage = total_count / total_count_master_mask    
            if i == MAX_RAND_ITER - 1 :
                # if this line evaluates, something is terribly wrong!
                print("Failure to converge area in prescribed iterations at time", t)
                sys.exit("Increase the number of hardcoded iterations!")

        # filling the whole month with the same generated mask
        for t2 in range( t, duration ):
            diff_mask=nan_mask_3D[t] ^ nan_mask_3D[t2]
            if diff_mask.sum() != 0:
                t = t2
                clear_flag = False

                current_time = time.time()

                if t != 0:
                    print( "Elapsed time:", round(current_time - time_zero), "s,  Estimated time:",
                          round((current_time - time_zero) * (duration / t - 1)), "s,  Total time:",
                          round((current_time - time_zero) * (duration / t)), "s" )

                break
            else:
                random_mask_3D[t2] = random_mask

        # Just to handle the clean exit after the last day in history
        if( clear_flag ):
            t2 += 1
            t = t2
    
    print("Finished random mask generation...")
    print("Maximum number of miss iterations =", max_miss_iter)
    
    training_mask_3D_bool = evaluation_mask_3D & (~random_mask_3D)
    
    return training_mask_3D_bool


def main(argv):
    percent_masked = 0.6
    eps_p = 0.005
    
    try:
        opts, args = getopt.getopt(argv,"hp:e:",["help","percent_masked=","eps_precision="])
    except getopt.GetoptError:
        print(sys.argv[0] + ' -p <percent_masked> -e <eps_precision>')
        print(sys.argv[0] + ' --help  for more options')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(sys.argv[0] + ' -p <percent_masked> -e <eps_precision>')
            print("Options:")
            print("  -p, --percent_masked=num   Specify ratio of masked data. Default is", percent_masked)
            print("  -e, --eps_precision=num    Specify precision of generated masked data ratio. default is", eps_p)
            sys.exit()
        elif opt in ("-p", "--percent_masked"):
            try:
                percent_masked = float(arg)
                print("Ratio of masked data set to", percent_masked)
            except ValueError:
                sys.exit("Bad argument.")
        elif opt in ("-e", "--eps_precision"):
            try:
                eps_p = float(arg)
                print("Precision set to", eps_p)
            except ValueError:
                sys.exit("Bad argument.")
    
    # Data files output
    data_dir = pathlib.Path('input_data')

    training_mask_3D_bool = generate_random_mask_history(data_dir, percent_masked, eps_p)

    print("Saving data to", data_dir)
    np.save(data_dir / ("training_mask_3D-v2-" + str(percent_masked) + "-" + str(eps_p)), training_mask_3D_bool)


if __name__ == "__main__":
   main(sys.argv[1:])
