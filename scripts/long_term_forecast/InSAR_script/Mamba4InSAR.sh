export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path_period ./dataset/Rovegliana_period_merged_data7 \
  --root_path_trend ./dataset/Rovegliana_trend_new \
  --data_path InSAR_1.csv \
  --model_id  Rovegliana_WD_MS_Mamba_dmdf256_el1_dl1_fc1_ex2_dc2_nh4_train \
  --model_period Mamba \
  --model_trend iTransformer \
  --data InSAR \
  --features_period MS \
  --features_trend S \
  --seq_len 60 \
  --label_len 60 \
  --pred_len 5 \
  --e_layers 1 \
  --d_layers 1 \
  --expand 2 \
  --d_conv 2 \
  --n_heads 4 \
  --factor 1 \
  --enc_in_period 8 \
  --dec_in_period 8 \
  --c_out_period 8 \
  --enc_in_trend 1 \
  --dec_in_trend 1 \
  --c_out_trend 1 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --itr 1 \
  --train_epochs 10 \
  --learning_rate_period 0.000696  \
  --learning_rate_trend 0.000696 \
  --batch_size 32 \
  --num_workers 5 \


