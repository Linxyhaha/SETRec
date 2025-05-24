dataset=${1}
lr=${2}
n_sem=${3}
n_query=$((n_sem + 1))
alpha=${4}
lora_module=("q_proj v_proj o_proj")
model_class=Qwen4Rec 
n_cf=1
seed=42
AE_layers="512 256 128"

suffix=SETRec-${lr}lr-${n_query}q-${n_sem}sem-${alpha}a-${AE}layers-${seed}seed
logfile=../log/qwen/${dataset}/${suffix}.log # make sure root path for log exists (../log/T5/${dataset})

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=12522 finetune_qwen.py \
    --base_model /your/path/to/qwen/model \
    --data_path ../data/${dataset}/ \
    --output_dir ../ckpt/ \
    --output_dir ../ckpt/qwen/${dataset}/${suffix} \
    --sem_encoder qwen \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules ${lora_module} \
    --n_query ${n_query} \
    --n_sem ${n_sem} \
    --n_cf ${n_cf} \
    --alpha ${alpha} \
    --layers ${AE_layers} \
    --batch_size 512 \
    --micro_batch_size 64 \
    --num_epochs 20 \
    --learning_rate ${lr} \
    --cutoff_len 4096 \
    --val_set_size 2000 \
    --lr_scheduler 'cosine' \
    --seed ${seed} \
    --warmup_steps 100 \
    > ${logfile}

# example
# bash scripts/train_qwen.sh toys 3e-4 6 0.5