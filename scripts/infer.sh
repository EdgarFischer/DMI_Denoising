# This script is used to actually apply a trained model to a new dataset and save the output

#!/bin/bash

# === Variablen ===
MODEL_NAME="fn_251009_WB_DMI_P01_pos4"
DATASET_NAME="fn_251009_WB_DMI_P01_4pos/output_4"

# === Aufruf ===
python infer.py \
  --config ../trained_models/${MODEL_NAME}/train.yaml \
  --ckpt ../trained_models/${MODEL_NAME}/checkpoints/last.pt \
  --input ../datasets/${DATASET_NAME}/CombinedCSI.mat \
  --output ../datasets/${DATASET_NAME}/CombinedCSIDenoised.mat \
  --batch-size 30 \
  --gpu 2