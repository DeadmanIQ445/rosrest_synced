#CUDA_VISIBLE_DEVICES=0 python train.py \
#  --config config/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
#  --augment \
#  --dataset_path /home/shamil/dataset_1024 \
#  --epochs 300 \
#  --batch-size 8 \
#  --lr 0.002

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config config/mask_rcnn_R_50_FPN_3x.yaml \
  --augment \
  --dataset_path /home/shamil/new_data/dataset \
  --epochs 100 \
  --batch-size 6 \
  --lr 0.002
#  --weights weights/aerial_summer_pieceofland.pth



