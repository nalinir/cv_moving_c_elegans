<<<<<<< HEAD
# ANTSUN

Implements the Automatic Neuron Tracking System for Unconstrained Nematodes. This notebook takes as input raw video data from NIR and confocal microscopes and outputs neural traces and behavioral parameters for that worm.

Version 1.3.5 is the version from [this article](https://github.com/flavell-lab/AtanasKim-Cell2023/tree/main#citation).

Version 2.1.0 is the version from [this article](https://doi.org/10.1101/2024.07.18.601886).

## Neural network weights

Most neural network weights should be available in [our Zenodo repository](https://zenodo.org/records/8185377). We've since updated the 3D U-Net segmentation network to include NeuroPAL training data; the updated weights are available [here](https://www.dropbox.com/scl/fo/zn530f0lnw9p8wqssqfwq/h?rlkey=01izs13oa9ef4hdw9ielhaqcx&dl=0).

## `elastix` parameters

Our `elastix` parameters are available in [our registration package](https://github.com/flavell-lab/RegistrationGraph.jl/tree/master/params).


## Citation

To cite this work, please refer to the following papers:

#### Brain-wide representations of behavior spanning multiple timescales and states in C. elegans
**Adam A. Atanas\***, **Jungsoo Kim\***, Ziyu Wang, Eric Bueno, McCoy Becker, Di Kang, 
Jungyeon Park, Talya S. Kramer, Flossie K. Wan, Saba Baskoylu, Ugur Dag,  Elpiniki Kalogeropoulou,
Matthew A. Gomes, Cassi Estrem, Netta Cohen, Vikash K. Mansinghka, Steven W. Flavell

Cell 2023; doi: https://doi.org/10.1016/j.cell.2023.07.035

**\* Equal Contribution**

#### Deep Neural Networks to Register and Annotate the Cells of the *C. elegans* Nervous System
Adam A. Atanas, Alicia Kun-Yang Lu, Jungsoo Kim, Saba Baskoylu, Di Kang, Talya S. Kramer, Eric Bueno, Flossie K. Wan, Steven W. Flavell

bioRxiv 2024; doi: https://doi.org/10.1101/2024.07.18.601886
=======
# cv_moving_c_elegans
>>>>>>> a6b124a (first commit)
