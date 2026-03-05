# This script is used to actually apply a trained model to a new dataset and save the output

python infer.py \
  --config configs/train.yaml \
  --ckpt trained_models/ABC/checkpoints/last.pt \
  --input /workspace/Denoising/datasets/sf_brain_DMI_HC_pilot_normalized/data.npy \
  --output /workspace/Denoising/datasets/sf_brain_DMI_HC_pilot_normalized_out/DENOISED.npy \
  --batch-size 600 \
  --gpu 2