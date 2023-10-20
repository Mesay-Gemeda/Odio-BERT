#!/bin/sh

#SBATCH --job-name=xd0-test
#SBATCH --clusters=arc
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:00
#SBATCH --partition=short
#SBATCH --output=outputs.out
#SBATCH --error=errors.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paul.rottger@oii.ox.ac.uk

# reset modules and load Python
#module purge
#module load Anaconda3/2020.11

# activate the right conda environment
#source activate $DATA/conda-envs/lrh-env

for dataset in spanish; do
    #for modelpath in $DATA/home/mesay/project_gpt/low-resource-hate/finetuned-models/random-sample/multilingual-models/xlm_dyn21_en_*${dataset}*/; do
    for modelpath in $DATA/home/mesay/project_gpt/low-resource-hate/finetuned-models/full-sample/multilingual-models/xlm-roberta-base_spanish_Spanish_full; do
        #for testpath in $DATA/home/mesay/project_gpt/low-resource-hate/0_data/main/old/1_clean/${dataset}/test_*.csv; do
        for testpath in $DATA/home/mesay/project_gpt/low-resource-hate/0_data/main/old/1_clean/spanish/spanish_hate_test.csv; do
            CUDA_VISISBLE_DEVICES=0,1 python /home/mesay/project_gpt/low-resource-hate/2_finetuning/finetune_and_test.py \
                --model_name_or_path ${modelpath} \
                --dataset_cache_dir $DATA/home/mesay/project_gpt/low-resource-hate/z_cache/datasets \
                --do_predict \
                --test_file ${testpath} \
                --test_results_dir $DATA/home/mesay/project_gpt/low-resource-hate/results/${dataset}_$(basename $testpath .csv) \
                --test_results_name $(basename $modelpath).csv \
                --per_device_eval_batch_size 16 \
                --max_seq_length 128 \
                --output_dir .
        done
    done
done