MPATH="MKP_LLaVA"
data_path="/path/to/datasets/train.json"
output_dir="/path/to/checkpoints/$MPATH"
mkdir -p $output_dir
deepspeed --include localhost:0,1,2,3 --master_port 28944 /path/to/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /path/to/scripts/zero2.json \
    --model_name_or_path /path/to/models/llava-v1.5-7b \
    --version v1 \
    --data_path $data_path\
    --eval_data_path /path/to/datasets/valid.json \
    --image_folder /path/to/datasets/CMKP_images \
    --vision_tower /path/to/models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_dir\
    --num_train_epochs 6\
    --per_device_train_batch_size 16\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --metric_for_best_model "sum"\
    --save_strategy "epoch" \
    --save_steps 0\
    --save_total_limit 10\
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --use_aimkp False 

# CUDA_VISIBLE_DEVICES=0,1,2,3  accelerate launch --num_processes 4 --main_process_port 25336 \
#     /path/to/evaluate.py --model-path $output_dir/best-checkpoint --model-base /path/to/models/llava-v1.5-7b \
#     --txt-path $MPATH-b5ds  --beam 5 --do-sample True