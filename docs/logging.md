# Logging

SuperSegmentaion provides lightweight logging for training and evaluation.

- **TensorBoard**: training scripts create TensorBoard writers that record losses, metrics and example images.
- **Checkpoints**: `utils/utils.py` supplies `save_checkpoint` and `load_checkpoint` for model states.
- **CSV Summaries**: functions like `append_csv` in `utils/utils.py` maintain experiment logs in comma-separated format.
- **Print Helpers**: `utils/print_tool.py` offers colored and structured console output.

Logs are stored under the `EXPER_PATH` specified in `settings.py`.

