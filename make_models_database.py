#!/usr/bin/env python
# coding: utf-8

# Load libraries

import pandas as pd
import pathlib
import glob
import sys
import getopt

from database_categories_spec import database_categories


def main(argv):
    input_database_name = None

    # Parsing command line arguments
    try:
        opts, args = getopt.getopt(argv, "hi:", [
                                   "help", "input_models_ensemble_name="])
    except getopt.GetoptError:
        print("Usage: ", sys.argv[0] + " --input_models_ensemble_name=Name")
        print(sys.argv[0] + " --help  for more options")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: ", sys.argv[0] + " --input_models_ensemble_name=Name")
            print("  -i Name,  --input_models_ensemble_name=Name    Name of input models ensemble in models directory")
            sys.exit()
        elif opt in ("-i", "--input_models_ensemble_name"):
            try:
                input_database_name = arg
                print("Input ensemble models name is set to", input_database_name)
            except ValueError:
                sys.exit("Bad argument.")

    assert input_database_name is not None  

    # Saving database with calculated models info
    input_database_name_path = pathlib.Path(input_database_name + ".csv")
    models_dir = pathlib.Path('models')
    input_database_full_path = str(
        models_dir / (input_database_name + ".num=*" + ".csv"))
    print("Reading input from multiple databases:", input_database_full_path)
    input_databases = [pd.read_csv(name)
                       for name in glob.glob(input_database_full_path)]
    if input_databases == []:
        sys.exit(
            "ERROR: Input databases do not exist.")
    input_database = pd.DataFrame(columns=database_categories)
    input_database = input_database.append(
        input_databases, ignore_index=True, sort=False)
    input_database.sort_values("model_num", inplace=True)
    print("Writing to single database file:", input_database_name_path)
    input_database.to_csv(input_database_name_path)



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
