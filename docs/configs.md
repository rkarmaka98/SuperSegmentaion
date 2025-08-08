# Configurations

Experiments are driven by YAML files in the `configs/` directory.

## Structure
Each configuration file defines:
- Dataset parameters (paths, augmentation choices).
- Model hyperparameters and checkpoints.
- Optimization settings such as learning rate and batch size.

## Usage
`train4.py` loads a base config and allows overrides via the command line. The final configuration is logged for reproducibility.

