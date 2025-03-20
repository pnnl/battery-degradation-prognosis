export CUDA_VISIBLE_DEVICES=2

root_path_name="./dataset/ROVI/"
checkpoint_path="./checkpoints/"
model_setting="ROVI_forecast_degradation4_"$1"_"$2"_iTransformer_1D_ftM_sl"$1"_ll0_pl"$2"_dm32_nh4_el3_dl3_df128_expand2_dc4_fc1_ebtimeF_dtTrue_lr0.001_do0.0_patience20_Exp_0"

if [ ! -d "$checkpoint_path/$model_setting" ]; then
  echo "No saved checkpoint for $1_$2"
  exit 1
fi


model_name=iTransformer

seq_len=$1
pred_len=$2

python -u predict_ROVI.py \
  --checkpoints $checkpoint_path \
  --saved_model $model_setting \
  --model_id $model_setting \
  --root_path $root_path_name \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --task_name ROVI_forecast \
  --data 1D \
  --features M \
  --label_len 0 \
  --is_training 0 \
  --inverse \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --e_layers 3 \
  --d_layers 3 \
  --n_heads 4 \
  --d_model 32 \
  --d_ff 128 


