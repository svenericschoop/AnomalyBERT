#!/bin/bash

# AnomalyBERT Custom Training Script
# This script automates the complete training process for custom unlabeled data

set -e  # Exit on any error

# Configuration
DATA_FILE="training_data_anomalyBERT.txt"
DATASET_NAME="CUSTOM"
GPU_ID=0
MAX_STEPS=100000
BATCH_SIZE=16
LEARNING_RATE=1e-4
SUMMARY_STEPS=1000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== AnomalyBERT Custom Training Setup ===${NC}"

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Error: Data file '$DATA_FILE' not found!${NC}"
    echo "Please ensure your training data file is in the current directory."
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found!${NC}"
    exit 1
fi

# Check if CUDA is available
echo -e "${YELLOW}Checking CUDA availability...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Step 1: Preprocess data
echo -e "${YELLOW}Step 1: Preprocessing data...${NC}"
python preprocess_custom_data.py \
    --input_file "$DATA_FILE" \
    --output_dir processed \
    --dataset_name "$DATASET_NAME"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Data preprocessing failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Data preprocessing completed successfully!${NC}"

# Step 2: Start training
echo -e "${YELLOW}Step 2: Starting training...${NC}"
echo "Training parameters:"
echo "  Dataset: $DATASET_NAME"
echo "  GPU ID: $GPU_ID"
echo "  Max steps: $MAX_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Summary steps: $SUMMARY_STEPS"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Start training with nohup for background execution
nohup python train_custom.py \
    --dataset "$DATASET_NAME" \
    --gpu_id "$GPU_ID" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --summary_steps "$SUMMARY_STEPS" \
    --n_features 512 \
    --patch_size 4 \
    --d_embed 512 \
    --n_layer 6 \
    --dropout 0.1 \
    --replacing_rate_max 0.15 \
    --soft_replacing 0.5 \
    --uniform_replacing 0.15 \
    --peak_noising 0.15 \
    --loss bce \
    > training.log 2>&1 &

TRAINING_PID=$!
echo -e "${GREEN}Training started with PID: $TRAINING_PID${NC}"
echo "Logs are being written to: training.log"
echo "Model checkpoints will be saved in: logs/"

# Provide monitoring commands
echo ""
echo -e "${YELLOW}Monitoring commands:${NC}"
echo "  View training progress: tail -f training.log"
echo "  Check GPU usage: nvidia-smi"
echo "  Stop training: kill $TRAINING_PID"
echo "  Start TensorBoard: tensorboard --logdir logs --port 6006"

echo ""
echo -e "${GREEN}Training is now running in the background!${NC}"
echo "Use 'tail -f training.log' to monitor progress."
