# This script is used to actually apply a trained model to a new dataset and save the output

python infer.py \
  --config ../configs/train.yaml \
  --ckpt ../trained_models/fn_260306_DMI_3T_invivo_WB/checkpoints/last.pt \
  --input ../datasets/fn_260306_DMI_3T_invivo_WB/CombinedCSI.mat \
  --output ../datasets/fn_260306_DMI_3T_invivo_WB/CombinedCSIDenoised.mat \
  --batch-size 600 \
  --gpu 2