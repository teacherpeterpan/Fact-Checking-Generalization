GPU=$1
MODEL_NAME=$2
MODEL=${MODEL_NAME//[\/]/_}
TRAIN_DATASETS=('FEVER-sentence' 'FEVER-paragraph' 'FAVIQ' 'FoolMeTwice' 'VitaminC' 'Climate-FEVER-paragraph' 'Climate-FEVER-sentence' 'Sci-Fact-paragraph' 'Sci-Fact-sentence' 'PUBHEALTH' 'COVID-Fact')
TEST_DATASETS=('Climate-FEVER-sentence' 'COVID-Fact' 'FAVIQ' 'FoolMeTwice' 'FEVER-paragraph' 'FEVER-sentence' 'VitaminC' 'Climate-FEVER-paragraph' 'Sci-Fact-paragraph' 'Sci-Fact-sentence' 'PUBHEALTH')

for((i=0;i<${#TRAIN_DATASETS[@]};i++)) do
    for((j=0;j<${#TEST_DATASETS[@]};j++)) do
        TRAIN=${TRAIN_DATASETS[i]}
        TEST=${TEST_DATASETS[j]}
        RUN_NAME=${TRAIN}-${MODEL}-${TEST}-binary
        echo "logs/${RUN_NAME}_pred.log"
        CUDA_VISIBLE_DEVICES=${GPU} python run_glue.py \
          --model_name_or_path model_save/${TRAIN}-${MODEL}-binary \
          --do_predict \
          --train_file data/${TEST}/train_binary.json \
          --validation_file data/${TEST}/dev_binary.json \
          --test_file data/${TEST}/dev_binary.json \
          --max_seq_length 200 \
          --per_device_eval_batch_size 512 \
          --overwrite_output_dir \
          --output_dir model_save/pred_output/${RUN_NAME} \
          > logs/${RUN_NAME}_pred.log 2>&1 
    done
done
