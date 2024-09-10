GPU=$1
MODEL=$2
TRAIN_DATASETS=('FEVER-paragraph' 'FEVER-sentence' 'VitaminC' 'Climate-FEVER-paragraph' 'Climate-FEVER-sentence' 'Sci-Fact-paragraph' 'Sci-Fact-sentence' 'PUBHEALTH')
TEST_DATASETS=('FEVER-paragraph' 'FEVER-sentence' 'VitaminC' 'Climate-FEVER-paragraph' 'Climate-FEVER-sentence' 'Sci-Fact-paragraph' 'Sci-Fact-sentence' 'PUBHEALTH')

for((i=0;i<${#TRAIN_DATASETS[@]};i++)) do
    for((j=0;j<${#TEST_DATASETS[@]};j++)) do
        TRAIN=${TRAIN_DATASETS[i]}
        TEST=${TEST_DATASETS[j]}
        RUN_NAME=${TRAIN}-${MODEL//[\/]/_}-${TEST}
        echo "logs/${RUN_NAME}_pred.log"
        CUDA_VISIBLE_DEVICES=${GPU} python run_glue.py \
            --model_name_or_path model_save/${TRAIN}-${MODEL} \
            --do_predict \
            --train_file data/${TEST}/train.json \
            --validation_file data/${TEST}/dev.json \
            --test_file data/${TEST}/dev.json \
            --max_seq_length 200 \
            --per_device_eval_batch_size 512 \
            --overwrite_output_dir \
            --output_dir model_save/pred_output/${RUN_NAME} \
            > logs/${RUN_NAME}_pred.log 2>&1 
    done
done
