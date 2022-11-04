export CUDA_VISIBLE_DEVICES=$1
RETRIEVED_CONTEXT=$2
OUTPUT_FILE=$3
allennlp evaluate output_with_evidence/model.tar.gz \
    ${RETRIEVED_CONTEXT} --output-file ${OUTPUT_FILE} --cuda-device 0 --include-package qasper_baselines
