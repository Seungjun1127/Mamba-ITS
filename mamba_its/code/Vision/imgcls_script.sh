#!/usr/bin/env bash

# CUDA 환경 변수 설정
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 GPU 아키텍처
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# PAM
# for dataset_prefix in differ_interpolation_-*0.5_**1_4*5_256*320_
# do
# CUDA_VISIBLE_DEVICES=0 python3 -m run_ImgCLS \
#     --model mambavision\
#     --seed 1799 \
#     --save_total_limit 1 \
#     --dataset PAM \
#     --dataset_prefix $dataset_prefix \
#     --train_batch_size 30 \
#     --eval_batch_size 30 \
#     --logging_steps 20 \
#     --save_steps 20 \
#     --epochs 20 \
#     --learning_rate 2e-5 \
#     --n_runs 1 \
#     --n_splits 5 \
#     --do_train
# done

#    --output_dir  "/root/Mamba-ITS/mamba_its/

for dataset_prefix in differ_interpolation_-*0.5_**1_4*5_256*320_
do
    # CUDA 캐시 정리
    rm -rf ~/.cache/torch
    python3 -c "import torch; torch.cuda.empty_cache()"

    # 스크립트 실행
    CUDA_VISIBLE_DEVICES=0 python3 -m run_ImgCLS \
        --withmissingratio WITHMISSINGRATIO \
        --feature_removal_level set \
        --model mambavisionB21K \
        --seed 1799 \
        --save_total_limit 1 \
        --dataset PAM \
        --dataset_prefix $dataset_prefix \
        --train_batch_size 4 \
        --eval_batch_size 8 \
        --logging_steps 20 \
        --save_steps 20 \
        --epochs 20 \
        --learning_rate 2e-5 \
        --n_runs 1 \
        --n_splits 5 \
        --gradient_checkpointing \
        --fp16 \
        --gradient_accumulation_steps 8 \
        --do_train
done 