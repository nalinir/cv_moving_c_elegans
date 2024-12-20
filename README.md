## Tracking neurons in (semi-)fixed _C. elegans_
This code accompanies our Computer Vision Final Project.

## Installation
Some packages installed better with conda, and some required pip installation. Given this, we have 2 separate requirements.txt files that can be downloaded separately.

## Baseline Model - Ultrack
We used Ultrack as our baseline model, the training output is available here:[INSERT LINK FROM RYAN]

## Data Preparation
We created our own pipeline for data preparation, which is available here: [INSERT LINK FROM MAREN]

## Model Training
We forked the Flavell lab's [BrainAlignNet](https://github.com/nalinir/BrainAlignNet/tree/main) code to make some key adjustments:
1. Adjusting hardcoding to reflect the dimensions of our new data
   * This includes adjusting their version of [DeepReg](https://github.com/nalinir/DeepReg/main) (which we also forked to track key changes)
2. Developing some registration code to reflect the new structure of our data, as well as address runtime issues experienced with the prior code

Given the size of the model and related output data, this code had to be run in .py files using SLURM jobs. To run this code, we need to run:
1. [demo_network.py](https://github.com/nalinir/BrainAlignNet/blob/main/scripts/demo_network.py)
2. [reg_matrix_fast.py](https://github.com/nalinir/BrainAlignNet/blob/main/scripts/reg_matrix_fast.py)

## Model Clustering
We removed some heuristics for our version of clustering as discussed in our paper. The final notebook is available here: [INSERT LINK FROM NALINI FINAL COMMIT]

## Evaluation Metrics and Visualizations
We have 2 notebooks for evaluation metrics and visualizations:
1. [evaluate.py](https://github.com/nalinir/cv_moving_c_elegans/blob/main/evaluate.ipynb) - metrics3 and metrics6 show the outputs for all worms using BrainAlignNet, as shown in the writeup 
2. Visualizations [FROM MAREN]