# This script is used to actually apply a trained model to a new dataset and save the output

python infer.py \
  --config ../configs/train.yaml \
  --ckpt ../trained_models/InVivo_22_22_21_96_8/checkpoints/last.pt \
  --input ../datasets/sf_brain_DMI_HC_pilot_normalized/CombinedCSI.mat \
  --output ../datasets/sf_brain_DMI_HC_pilot_normalized/CombinedCSIDenoised.mat \
  --batch-size 600 \
  --gpu 2