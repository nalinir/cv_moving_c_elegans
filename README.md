## Tracking neurons in (semi-)fixed _C. elegans_
This code accompanies our Computer Vision Final Project. The goal is to run BrainAlignNet ([Atanas et al., 2024]([https://www.genome.gov/](https://www.biorxiv.org/content/biorxiv/early/2024/07/22/2024.07.18.601886.full.pdf))) as part of a new pipeline for neuron tracking, with calcium imaging data and adapted pre-processing of the BrainAlignNet inputs and post-processing of its outputs for tracking. For segmentation, we used StarDist ([Weigert et al., 2020]([https://www.genome.gov/](https://openaccess.thecvf.com/content_WACV_2020/html/Weigert_Star-convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.html))), and as a baseline, we used Ultrack ([Bragantini et al., 2024]([https://www.genome.gov/](https://www.biorxiv.org/content/biorxiv/early/2024/07/22/2024.07.18.601886.full.pdf](https://pmc.ncbi.nlm.nih.gov/articles/PMC11398427/)).

## Installation
Some packages installed better with conda, and some required pip installation. Given this, we have 2 separate requirements.txt files that can be downloaded separately. These are available in the [requirements](https://github.com/nalinir/cv_moving_c_elegans/tree/main/requirements) folder.

## Baseline Model - Ultrack
We used Ultrack as our baseline model, the training output is available here: [Ultrack_Code](https://github.com/nalinir/cv_moving_c_elegans/tree/main/Ultrack_Baseline)

## Data Preparation
We created our own pipeline for data preparation, which is available here: [preprocess_data.ipynb](https://github.com/nalinir/cv_moving_c_elegans/preprocess_data.ipynb)

## Model Training
We forked the Flavell lab's [BrainAlignNet](https://github.com/nalinir/BrainAlignNet/tree/main) code to track some key adjustments:
1. Adjusting hardcoding to reflect the dimensions of our new data
   * This includes adjusting their version of [DeepReg](https://github.com/nalinir/DeepReg/main) (which we also forked to track key changes)
2. Developing some registration code to reflect the new structure of our data, as well as address runtime issues experienced with the prior code

Given the size of the model and related output data, this code had to be run in .py files using SLURM jobs. To run this code, we need to run:
1. [demo_network.py](https://github.com/nalinir/BrainAlignNet/blob/main/scripts/demo_network.py)
2. [reg_matrix_fast.py](https://github.com/nalinir/BrainAlignNet/blob/main/scripts/reg_matrix_fast.py)

## Model Clustering
We removed some heuristics for our version of clustering as discussed in our paper. The final notebook is available here: [clustering.ipynb](https://github.com/nalinir/cv_moving_c_elegans/blob/main/clustering.ipynb)

## Evaluation Metrics and Visualizations
We have 2 notebooks for evaluation metrics and visualizations:
1. [evaluate.py](https://github.com/nalinir/cv_moving_c_elegans/blob/main/evaluate.ipynb) - metrics3 and metrics6 show the outputs for all worms using BrainAlignNet, as shown in the writeup 
2. Visualizations [Figures.ipynb](https://github.com/nalinir/cv_moving_c_elegans/Figures.ipynb)

## Future Work
We have developed the ROI heuristic, as mentioned in the paper, for future training implementations
1. [reg_matrix_with_heuristic.ipynb](https://github.com/nalinir/BrainAlignNet/blob/main/scripts/reg_matrix_with_heuristic.ipynb)
