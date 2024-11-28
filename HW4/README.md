# CPSC 8430 Homework 4

This folder contains all the scripts/python files used to generate the report **HW4_report_jkbrook**

## How to run the model
Click on the link provided in ***access_models.txt*** and download the required folder(s).

All 3 models were run using the batch scripts ***run_{model_name}.sh*** on Clemson's Palmetto Cluster. Prior to running the script, please ensure all file paths are updated to include your own working directory.

This script can be submitted using the following command:
```
  # make sure you are in your working directory
  cd </path/to/working/directory>

  sbatch scripts/run_{model_name}.sh
```
Loss comparison was done using the jupyter notebook file ***loss.ipynb***
