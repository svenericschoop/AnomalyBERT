# AnomalyBERT Custom Training Deployment Guide

This guide provides step-by-step instructions for training AnomalyBERT with your custom unlabeled data on a remote server.

## Prerequisites

1. **Python Environment**: Python 3.8+ with pip
2. **CUDA**: NVIDIA GPU with CUDA support (recommended)
3. **Storage**: Sufficient disk space for data and model checkpoints

## Step 1: Prepare Your Data

1. **Upload your data file** to the server:
   ```bash
   scp training_data_anomalyBERT.txt user@your-server:/path/to/AnomalyBERT/
   ```

2. **Verify data format**: Your data should be in CSV format with comma-separated values, similar to the SMD dataset format. The data should already be scaled between 0 and 1.

## Step 2: Set Up Environment

1. **Clone or upload the AnomalyBERT code** to your server:
   ```bash
   git clone <your-repo-url> /path/to/AnomalyBERT
   cd /path/to/AnomalyBERT
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU availability** (if using GPU):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Step 3: Preprocess Your Data

1. **Run the preprocessing script**:
   ```bash
   python preprocess_custom_data.py \
       --input_file training_data_anomalyBERT.txt \
       --output_dir processed \
       --dataset_name CUSTOM
   ```

   This will create:
   - `processed/CUSTOM_train.npy` - Your training data (already scaled 0-1)
   - `processed/CUSTOM_test.npy` - Dummy test data (subset of training)
   - `processed/CUSTOM_test_label.npy` - Dummy test labels (all zeros)

2. **Verify preprocessing**:
   ```bash
   ls -la processed/CUSTOM_*
   ```

## Step 4: Configure Training Parameters

1. **Check your data dimensions**:
   ```bash
   python -c "import numpy as np; data = np.load('processed/CUSTOM_train.npy'); print(f'Shape: {data.shape}, Range: [{data.min():.3f}, {data.max():.3f}]')"
   ```

2. **Update config if needed**: If your data has a different number of columns than 10, edit `utils/config.py`:
   ```python
   NUMERICAL_COLUMNS['CUSTOM'] = tuple(range(YOUR_NUM_COLUMNS))
   ```

## Step 5: Start Training

1. **Basic training command**:
   ```bash
   python train_custom.py \
       --dataset CUSTOM \
       --gpu_id 0 \
       --max_steps 50000 \
       --batch_size 16 \
       --lr 1e-4
   ```

2. **Advanced training with custom parameters**:
   ```bash
   python train_custom.py \
       --dataset CUSTOM \
       --gpu_id 0 \
       --max_steps 100000 \
       --batch_size 32 \
       --lr 5e-5 \
       --n_features 256 \
       --patch_size 8 \
       --d_embed 256 \
       --n_layer 4 \
       --dropout 0.1 \
       --replacing_rate_max 0.2 \
       --soft_replacing 0.6 \
       --uniform_replacing 0.2 \
       --peak_noising 0.15 \
       --summary_steps 1000
   ```

## Step 6: Monitor Training

1. **Check training logs**:
   ```bash
   # View latest log directory
   ls -la logs/
   
   # Monitor training progress
   tail -f logs/[timestamp]_CUSTOM/events.out.tfevents.*
   ```

2. **Use TensorBoard** (optional):
   ```bash
   tensorboard --logdir logs --port 6006
   # Access via http://your-server:6006
   ```

## Step 7: Save and Use Trained Model

1. **Model checkpoints** are saved in:
   - `logs/[timestamp]_CUSTOM/model.pt` - Full model
   - `logs/[timestamp]_CUSTOM/state/state_dict_step_*.pt` - Training checkpoints
   - `logs/[timestamp]_CUSTOM/state_dict.pt` - Final model state

2. **Load trained model** for inference:
   ```python
   import torch
   from models.anomaly_transformer import get_anomaly_transformer
   
   # Load model
   model = torch.load('logs/[timestamp]_CUSTOM/model.pt')
   model.eval()
   
   # Use for inference
   # ... your inference code here
   ```

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**:
   - Reduce `--batch_size` (try 8 or 4)
   - Reduce `--n_features` (try 256 or 128)
   - Reduce `--d_embed` (try 256 or 128)

2. **Data loading errors**:
   - Check data format (should be CSV with comma separators)
   - Verify file paths in config.py
   - Ensure data is properly normalized

3. **Training too slow**:
   - Increase `--batch_size` if memory allows
   - Reduce `--n_features` and `--d_embed`
   - Use fewer transformer layers (`--n_layer`)

### Performance Tips:

1. **For large datasets**: Use `--max_steps 200000` or more
2. **For small datasets**: Use `--max_steps 50000` or less
3. **Memory optimization**: Start with smaller parameters and scale up
4. **Training time**: Expect 1-3 hours for 50k steps on a modern GPU

## Example Complete Workflow

```bash
# 1. Setup
cd /path/to/AnomalyBERT
pip install -r requirements.txt

# 2. Preprocess data
python preprocess_custom_data.py \
    --input_file training_data_anomalyBERT.txt \
    --dataset_name CUSTOM

# 3. Train model
python train_custom.py \
    --dataset CUSTOM \
    --gpu_id 0 \
    --max_steps 100000 \
    --batch_size 16 \
    --lr 1e-4 \
    --summary_steps 1000

# 4. Monitor (in another terminal)
tensorboard --logdir logs --port 6006
```

## Notes

- The training uses unsupervised learning, so no ground truth labels are needed
- The model learns to detect anomalies by reconstructing normal patterns
- Training time depends on data size, model complexity, and hardware
- Save checkpoints regularly for long training runs
- Use `nohup` for background training: `nohup python train_custom.py ... &`
