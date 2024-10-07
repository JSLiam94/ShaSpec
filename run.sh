#!/usr/bin/env bash
time=$(date "+%Y%m%d-%H%M%S")
name=BraTS24_ShaSpec

CUDA_VISIBLE_DEVICES=$1 python train_SS.py \
--snapshot_dir=snapshots/$name/ \
--input_size=80,160,160 \
--batch_size=1 \
--num_gpus=1 \
--num_steps=80000 \
--val_pred_every=10000 \
--learning_rate=1e-2 \
--num_classes=3 \
--num_workers=4 \
--train_list=train.csv \
--val_list=val.csv \
--random_mirror=True \
--random_scale=True \
--weight_std=True \
--reload_path=snapshots/BraTS24_ShaSpec1/last.pth \
--reload_from_checkpoint=True \
--mode=random #> logs/${time}_train_${name}.log
