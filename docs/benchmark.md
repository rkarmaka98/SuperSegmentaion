# Benchmarking

Benchmark scripts allow comparison of SuperSegmentaion with other feature extractors.

- **HPatches**: `evaluations/` modules follow the HPatches protocol for repeatability, homography accuracy and descriptor matching.
- **Classical Baselines**: `export_classical.py` exports results for SIFT/ORB enabling side-by-side evaluation.
- **Summary Utilities**: `summarize.py` and `summary.csv` collate scores across datasets and checkpoints.

Results can be reproduced by running `export.py` followed by `evaluation.py` on the generated `.npz` pairs.

