#!/bin/sh

#SBATCH --job-name=ft2x-hi-full
#SBATCH --clusters=arc
#SBATCH --ntasks-per-node=16
#SBATCH --time=11:59:00
#SBATCH --partition=short
#SBATCH --output=outputs.out
#SBATCH --error=errors.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk

# reset modules
#module purge

# load python module
#module load Anaconda3/2020.11

# activate the right conda environment
#source activate $DATA/conda-envs/lrh-env

# Pick base model to then continue finetuning (-->FT2)
# basemodel="xlm_dyn21_en_20000_rs1"

# for dataset in orm; do
#     CUDA_VISISBLE_DEVICES=0,1 python /home/mesay/project_gpt/low-resource-hate/2_finetuning/finetune_and_test.py \
#         --model_name_or_path $DATA/home/mesay/project_gpt/low-resource-hate/english-base-models/${basemodel}/ \
#         --train_file $DATA/home/mesay/project_gpt/low-resource-hate/0_data/main/old/1_clean/${dataset}/train_*.csv \
#         --validation_file $DATA/home/mesay/project_gpt/low-resource-hate/0_data/main/old/1_clean/${dataset}/dev_*.csv \
#         --dataset_cache_dir $DATA/home/mesay/project_gpt/low-resource-hate/z_cache/datasets \
#         --do_train \
#         --per_device_train_batch_size 4 \
#         --num_train_epochs 5 \
#         --max_seq_length 512 \
#         --do_eval \
#         --per_device_eval_batch_size 4 \
#         --evaluation_strategy "epoch" \
#         --save_strategy "epoch" \
#         --save_total_limit 1 \
#         --load_best_model_at_end \
#         --metric_for_best_model "macro_f1" \
#         --output_dir $DATA/home/mesay/project_gpt/low-resource-hate/finetuned-models/full-sample/multilingual-models/${basemodel}_${dataset}_10812_full \
#         --overwrite_output_dir

#     rm -rf $DATA/home/mesay/project_gpt/low-resource-hate/finetuned-models/full-sample/multilingual-models/${basemodel}_${dataset}_10812_full/check*
# done


basemodel="xlm-roberta-base"

for dataset in spanish; do
    CUDA_VISISBLE_DEVICES=0,1 python /home/mesay/project_gpt/low-resource-hate/2_finetuning/finetune_and_test.py \
        --model_name_or_path $DATA/home/mesay/project_gpt/low-resource-hate/default-models/${basemodel}/ \
        --train_file $DATA/home/mesay/project_gpt/low-resource-hate/0_data/main/old/1_clean/${dataset}/spanish_hate_train.csv \
        --validation_file $DATA/home/mesay/project_gpt/low-resource-hate/0_data/main/old/1_clean/${dataset}/spanish_hate_dev.csv\
        --dataset_cache_dir $DATA/home/mesay/project_gpt/low-resource-hate/z_cache/datasets \
        --do_train \
        --per_device_train_batch_size 16 \
        --num_train_epochs 5 \
        --max_seq_length 512 \
        --do_eval \
        --per_device_eval_batch_size 16 \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --load_best_model_at_end \
        --metric_for_best_model "macro_f1" \
        --output_dir $DATA/home/mesay/project_gpt/low-resource-hate/finetuned-models/full-sample/multilingual-models/${basemodel}_${dataset}_Spanish_full \
        --overwrite_output_dir

    rm -rf $DATA/home/mesay/project_gpt/low-resource-hate/finetuned-models/full-sample/multilingual-models/${basemodel}_${dataset}_Spanish_full/check*
done


