## Tracking neurons in (semi-)fixed _C. elegans_
Here is the code we used for tracking neurons in semi-fixed _C. elegans_.

1. Ultrack Model (Ryan and Maren)
2. Preprocessing for BrainAlignNet (Ryan and Maren)
3. [BrainAlignNet](https://github.com/nalinir/BrainAlignNet) (Nalini and Maren)
   * Repo is forked to track main changes (primarily hardcoding issues)
   * It also includes the experiments we ran (TBD if we want to include this)
   * We added the _registration.txt_ file, which was missing from the original BrainAlignNet code and had to be obtained via trial-and-error (Nalini to run this***)
5. Registration post-model run (Neel, Nalini, and Maren)
   * Making registration matrices
   * Clustering
   * Applying clustering
6. Evaluation (Ryan)
   * Provides evaluation for both models
