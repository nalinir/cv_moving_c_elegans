# ANTSUN

Implements the Automatic Neuron Tracking System for Unconstrained Nematodes. This notebook takes as input raw video data from NIR and confocal microscopes and outputs neural traces and behavioral parameters for that worm.

Version 1.3.5 is the version from [this article](https://github.com/flavell-lab/AtanasKim-Cell2023/tree/main#citation).

Version 2.1.0 is the version from the upcoming biorxiv article (Atanas et al., 2024).

## Neural network weights

Most neural network weights should be available in [our Zenodo repository](https://zenodo.org/records/8185377). We've since updated the 3D U-Net segmentation network to include NeuroPAL training data; the updated weights are available [here](https://www.dropbox.com/scl/fo/zn530f0lnw9p8wqssqfwq/h?rlkey=01izs13oa9ef4hdw9ielhaqcx&dl=0).

## `elastix` parameters

Our `elastix` parameters are available in [our registration package](https://github.com/flavell-lab/RegistrationGraph.jl/tree/master/params)

## Citation
To cite this work, please refer to [this article](https://github.com/flavell-lab/AtanasKim-Cell2023/tree/main#citation).
