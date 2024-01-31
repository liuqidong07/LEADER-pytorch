date="0105-online"
gpu_id=0
peft_path="./saved/lora-0104/checkpoint-4000/"
train_file="0104"

mark_name="online-KD"
python main_distill.py --dataset mimic4 \
                       --model_name pnet \
                       --train_file $train_file \
                       --train_batch_size 16 \
                       --log \
                       --gpu_id $gpu_id \
                       --num_train_epochs 100 \
                       --distill \
                       --check_path distill-$date \
                       --peft_path $peft_path \
                       --alpha 0.1 \
                       --d_loss mse \
                       --max_source_length 2048 \
                       --profile \
                       --num_workers 8 \
                       --align \
                       --align_weight 0.01 \
                       --mark_name $mark_name



