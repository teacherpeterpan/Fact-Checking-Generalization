GPU=$1
MODEL=$2
DATASETS=('FAVIQ' 'FoolMeTwice' 'FEVER-paragraph' 'FEVER-sentence' 'VitaminC' 'Climate-FEVER-paragraph' 'Climate-FEVER-sentence' 'Sci-Fact-paragraph' 'Sci-Fact-sentence' 'PUBHEALTH' 'COVID-Fact')

mkdir -p logs
mkdir -p model_save

for DATASET in ${DATASETS[@]}
do
    RUN_NAME=${DATASET}-${MODEL}-binary
    echo "logs/${RUN_NAME}.log"
    CUDA_VISIBLE_DEVICES=${GPU} python run_glue.py \
      --model_name_or_path ${MODEL} \
      --do_train \
      --do_eval \
      --do_predict \
      --train_file data/${DATASET}/train_binary.json \
      --validation_file data/${DATASET}/dev_binary.json \
      --test_file data/${DATASET}/dev_binary.json \
      --max_seq_length 200 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --gradient_accumulation_steps 2 \
      --num_train_epochs 5.0 \
      --learning_rate 1e-5 \
      --save_strategy epoch \
      --load_best_model_at_end \
      --metric_for_best_model marco-f1 \
      --overwrite_output_dir \
      --evaluation_strategy epoch \
      --output_dir model_save/${RUN_NAME} \
      --report_to none \
      > logs/${RUN_NAME}.log 2>&1 
    rm -r model_save/${RUN_NAME}/checkpoint-*/
 done
