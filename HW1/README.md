# CPSC 8430 Homework 1

This folder contains all the scripts/notebook/python files used to generate the report **HW1_report_jkbrook**

This homework was split into three parts and the files that complete them are as such:
- ***1-1: Deep vs. Shallow***
  - HW1_1-1.ipynb 
- ***1-2: Optimization***
  - HW1_1-2_PCA.py
  - HW1_1-2_GradNorm.py
- ***1-3: Generalization***
  - HW1_1-3_RandLabel.py
  - HW1_1-3_NumParams.py
  - HW1_1-3_FvG_train.py
  - HW1_1-3_FvG_sens.py

## How to run the files:

A conda environment containing the packages pytorch, matplotlib, and other necessary packages was utilized and will be needed when running the files.

For part 1-1 the Jupyter Notebook can be run using the Jupyter application.

For all other parts, batch scripts were written for the files to be submitted on Palmetto. 
The scipts can be submitted using the following commands:
```
  # Change directories into the Scripts folder
  cd </path/to/working/directory>/Scripts

  sbatch <desired .sh file>

  # If running on local environment
  python3 <desired python file>
```
