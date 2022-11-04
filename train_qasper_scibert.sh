export CUDA_VISIBLE_DEVICES=3,4,5,6

python -u -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.1 --master_port 9520 main.py \
	--my_config ./config.json \
	--model_name_or_path allenai/scibert_scivocab_uncased \
	--train_pattern ./data/Qasper/qasper-train-v0.2.json \
	--feature_path ./data/Qasper \
	--test_pattern ./data/Qasper/qasper-dev-v0.2.json \
	--output_dir ./output-scibert-qasper \
	--do_train --train_batch_size 1 \
	--do_predict  --predict_batch_size 1 \
	--learning_rate 1e-5 --warmup_proportion 0.1 --num_train_epochs 8 \
	--weight_decay 0.01 \
	--save_epochs 1 \
	--num_hidden_layers 1 \
	--neighborhops 4 \
	--meta_gat_hops 1 \
	--max_epochs 3 \
	--requires_grad 5,6,7,8,9,10,11,pooler \
	--indoc_num 16