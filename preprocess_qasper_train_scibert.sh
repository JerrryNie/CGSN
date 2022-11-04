INPUT_PATH=./data/Qasper/qasper-train-v0.2.json

python create_qasper_examples_scibert.py \
	--input_path $INPUT_PATH \
	--model_name_or_path allenai/scibert_scivocab_uncased \
	--do_lower_case \
	--is_training \
	--output_dir ./data/Qasper/ \
	--num_threads 24 --max_seq_length 256 --doc_stride 256 \
	--my_config ./config.json