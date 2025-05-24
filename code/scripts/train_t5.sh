dataset=${1}
lr=${2}
n_sem=${3}
n_query=$((n_sem + 1))
alpha=${4}
n_cf=1
seed=42
AE_layers="512 256 128"

suffix=SETRec-${lr}lr-${n_query}q-${n_sem}sem-${alpha}a-${AE}layers-${seed}seed
logfile=../log/t5/${dataset}/${suffix}.log # make sure root path for log exists (../log/T5/${dataset})

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12522 finetune_t5.py \
    --base_model /storage/xylin/LLM4Rec/identifier/code/instruments/E4SRec/t5-small \
    --data_path ../data/${dataset}/ \
    --output_dir ../ckpt/t5/ \
    --sem_encoder t5 \
    --n_query ${n_query} \
    --n_sem ${n_sem} \
    --n_cf ${n_cf} \
    --alpha ${alpha} \
    --layers ${AE_layers} \
    --batch_size 512 \
    --micro_batch_size 128 \
    --num_epochs 30 \
    --learning_rate ${lr} \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lr_scheduler 'cosine' \
    --seed ${seed} \
    --warmup_steps 100 \
    > ${logfile}

# example
# bash scripts/train_t5.sh toys 1e-3 4 0.7