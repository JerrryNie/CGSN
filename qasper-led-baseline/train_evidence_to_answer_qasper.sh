export CUDA_VISIBLE_DEVICES=$1

allennlp train training_config/led_base_smaller_context.jsonnet -s output_with_evidence --include-package qasper_baselines
