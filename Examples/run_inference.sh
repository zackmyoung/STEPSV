#!/usr/bin/env bash
set -euo pipefail

python3 STEPSV_v1.00.py \
  --mode inference \
  --data_dir "./GPS_data" \
  --output_dir "./example_inference_out" \
  --model "./STEPSV_final_model_epoch_070.pt" \
  --scaler_path "./STEPSV_scaler.pkl" \
  --apply_scaling 1 \
  --max_length 5500 \
  --pad_start 0 \
  --pad_end 0 \
  --linear_size 16 \
  --hidden_size 32 \
  --num_layers 1 \
  --dropout 0 \
  --bidirectional 1 \
  --mc_samples 0 \
  --infer_batch_size 128 \
  --infer_workers 4
