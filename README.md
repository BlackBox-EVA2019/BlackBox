# BlackBox

Our code was tested on Ubuntu 18.04 with CUDA 10.2, quad core system with NVIDIA GeForce RTX 2070 8 GB RAM GPU, and 48 GB system RAM.
It needs at least 20 GB of free disk space.

Run all code within the same directory containing `BlackBox-production.yml`, `DATA_TRAINING.RData` and `TRUE_DATA_RANKING.RData`.

Install `conda`: https://docs.conda.io/projects/conda/en/latest/user-guide/install/#regular-installation

Create conda environment with Python and R packages.

```shell
$ conda env create -f BlackBox-production.yml

$ conda activate BlackBox-production
```

Convert data and generate random training masks.
```shell
$ python convert_data.py

$ python generate_extra_masks.py  # takes up to 12 hours to complete

$ python shift_bitmaps.py

$ python downsample_bitmaps.py
```

Train models and evaluate their ensemble prediction (approximately 7 days to complete):
```shell
$ bash train_models.sh  # BASH script training an ensemble of 155 models, each with specific hyperparameters

$ python make_models_database.py -i Model

$ Rscript preprocesing_TRUE_DATA_RANKING.R  # creates file true.observations.rda with data used in score calculation

$ python make_prediction.py -i Model -o Model_score --number_of_predictions=20 --prediction_suffix=SampleA

$ python make_prediction.py -i Model -o Model_ensemble_score --only_score=20:SampleA --global_ensemble
```

Final prediction is saved in `./predictions/`. Calculated score is saved in the last line of `./Model_ensemble_score_20_SampleA.csv`.

# Comment on reproducibility
In our code we use the following Python libraries which rely on different random number generators: `random` from stdlib, `numpy`, `pytorch`, and `fastai`. Since at least `pytorch` [does not guarantee reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) between different platforms, CPU/GPU runs, and library versions even with a fixed random seed, we have opted not to seed any of the random number generators in our solution. This means a certain amount of variation in final result is expected between runs.
