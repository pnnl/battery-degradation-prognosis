export CUDA_VISIBLE_DEVICES=3

model_name=iTransformer

root_path_name="/Users/grac833/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/ROVI/code/PatchTST/PatchTST_supervised/cycles4/"

for lr in 0.0001
do

  for do in 0.05
  do

    echo $lr $do

    for iter in 5
    do

      python -u run.py \
        --task_name regression \
        --is_training 1 \
        --root_path $root_path_name \
        --model_id test_reg \
        --model $model_name \
        --data SOHReg \
        --features M \
        --e_layers 3 \
        --batch_size 64 \
        --d_model 128 \
        --d_ff 256 \
        --top_k 3 \
        --dropout $do\
        --fc_dropout $do\
        --head_dropout 0\
        --des 'Exp' \
        --itr $iter \
        --num_workers 1\
        --patch_len 32\
        --stride 1\
        --learning_rate $lr \
        --lradj constant \
        --train_epochs 100 \
        --patience 30
      done
  done
done