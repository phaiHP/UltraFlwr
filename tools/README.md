# Tools Directory

Development tools for baseline comparisons and analysis.

## Directory Structure

### `centralized/`
Centralized (non-federated) training for baseline comparisons:
- `local_train_and_test.py` - Simple centralized YOLO training
- `local_test_only.py` - Testing only
- `organize_log_results_local_train.py` - Parse and organize training logs
- `summarize_eval_results.py` - Summarize evaluation results
- `run_local_train_and_test.sh` - Shell wrapper for centralized training
- `run_local_eval.sh` - Shell wrapper for evaluation

## Usage

### Centralized Training (Baseline)
Run centralized training to compare against federated learning results:

```bash
./tools/centralized/run_local_train_and_test.sh
```

## Dataset Conversion

Dataset format converters are now part of the `fedyolo` package:

```python
# Convert COCO format to YOLO
python -m fedyolo.data.converters.endoscapes_coco_to_yolo

# Convert VOC format to YOLO
python -m fedyolo.data.converters.m2cai16_voc_to_yolo
```

## Federated Learning Workflows

For federated learning (the main functionality), use the Python CLI:

```bash
# Install package
pip install -e .

# Partition datasets
fedyolo-partition

# Run federated training
fedyolo-train

# Run comprehensive testing
fedyolo-test
```

See the main [README.md](../README.md) for full documentation.
