export CUDA_VISIBLE_DEVICES=2

root_path_name="/Users/grac833/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/ROVI/code/PatchTST/PatchTST_supervised/degradation3/"
#root_path_name="/Users/grac833/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/ROVI/code/PatchTST/PatchTST_supervised/cycles4/"
root_path_name="/Users/grac833/Emily_Documents/Projects/ROVI/data/degradation4/"


data_path_name=rovi2.csv

model_name=iTransformer

total=2000
#for seq_len in 50 150 250 350 450
for seq_len in 375
do

pred_len=$((total-seq_len))

echo $seq_len $pred_len
pred_len=2000


for lr in 0.001
do

  for do in 0.0
  do

    echo $lr $do

    for iter in 1
    do

      python -u run_ROVI.py \
        --task_name ROVI_forecast \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id 'degradation4_'$seq_len'_'$pred_len \
        --model $model_name \
        --data 1D \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --e_layers 3 \
        --d_layers 3 \
        --n_heads 4 \
        --d_model 32 \
        --d_ff 128 \
        --dropout $do\
        --des 'Exp' \
        --train_epochs 200\
        --target 'param'\
        --patience 20\
        --num_workers 1\
        --lradj constant\
        --fc_dropout $do\
        --head_dropout 0\
        --patch_len 32\
        --stride 4\
        --itr $iter --batch_size 64 --learning_rate $lr

    done
  done
done

done