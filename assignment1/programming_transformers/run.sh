dataset=lm_synthetic 
epochs=300 
lr=0.0005
train_batch=32

python src/experiments.py \
    --task ${dataset} \
    --epochs ${epochs} \
    --learning_rate ${lr} \
    --train_batch ${train_batch} \