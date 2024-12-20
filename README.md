## Tracking neurons in (semi-)fixed _C. elegans_
Here is the code we used for tracking neurons in semi-fixed _C. elegans_.

1. Ultrack Model (Ryan and Maren)
2. Preprocessing for BrainAlignNet (Ryan and Maren)
3. [BrainAlignNet](https://github.com/nalinir/BrainAlignNet) (Nalini and Maren)
   * Repo is forked to track main changes (primarily hardcoding issues)
   * It also includes the experiments we ran (TBD if we want to include this)
   * ***Add some notes on files run (this could also be in the forked readme)
5. Registration post-model run (Neel, Nalini, and Maren)
   * Making registration matrices
   * Clustering
   * Applying clustering
6. Evaluation (Ryan)
   * Provides evaluation for both models
  
Some packages installed better with conda, and some required pip installation. Given this, we have 2 separate requirements.txt files.
