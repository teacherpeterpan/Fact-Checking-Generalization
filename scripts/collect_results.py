import os
import json
import heapq
import pandas as pd
import argparse
from sklearn.metrics import classification_report, confusion_matrix

TRANSFER = 'transfer'
BINARY = 'binary'
SCRATCH = 'From Scratch'
MACRO_F1 = 'macro-f1'
CLASS_F1 = 'class-f1'
ACCU_F1 = 'accu-f1'

all_datasets = ['FEVER-paragraph', 'FEVER-sentence', 'VitaminC', 'Climate-FEVER-paragraph', 'Climate-FEVER-sentence', 'Sci-Fact-paragraph', 'Sci-Fact-sentence', 'PUBHEALTH']
all_binary_datasets = all_datasets[:3] + ['FoolMeTwice'] + all_datasets[3:] + ['COVID-Fact', 'FAVIQ']
all_exps = [TRANSFER, BINARY]
all_metrics = [MACRO_F1, CLASS_F1, ACCU_F1]

parser = argparse.ArgumentParser()
#parser.add_argument("--datasets", help="multi dataset names", nargs='*', default=all_datasets)
parser.add_argument("--model", help="huggingface model name", default='roberta-large')
parser.add_argument("--exp", help="RQ name", default=TRANSFER)
parser.add_argument("--metric", help="report f1 score for each class", default=MACRO_F1)
parser.add_argument("--highlight", help="bold highest number", default=0)
args = parser.parse_args()
MODEL = args.model
exp = args.exp
metric = args.metric
highlight = args.highlight

assert exp in all_exps
assert metric in all_metrics
dataset_namemap = {
        'FEVER-paragraph' : 'FEVER-para',
        'FEVER-sentence' : 'FEVER-sent',
        'Climate-FEVER-paragraph' : 'C-FEVER-para',
        'Climate-FEVER-sentence' : 'C-FEVER-sent',
        'Sci-Fact-paragraph' : 'SciFact-para',
        'Sci-Fact-sentence' : 'SciFact-sent',
        }
MODEL = MODEL.replace('/', '_')

all_results = []
if exp == BINARY:
    train_datasets = all_binary_datasets
    test_datasets = all_binary_datasets
else:
    train_datasets = all_datasets
    test_datasets = all_datasets

report_file_name = f"model_save/pred_output/results/{MODEL}_{exp}_{metric}_reports.txt"
report_writer = open(report_file_name, 'w')
columns = ["Train$\\downarrow$ Test$\\rightarrow$"] + [dataset_namemap.get(d, d) for d in test_datasets]
for TRAIN in train_datasets:
    row_data = [dataset_namemap.get(TRAIN, TRAIN)]
    for TEST in test_datasets:
        if exp == TRANSFER:
            RUN_NAME = f"{TRAIN}-{MODEL}-{TEST}"
        elif exp == BINARY:
            RUN_NAME = f"{TRAIN}-{MODEL}-{TEST}-binary"
        else:
            raise ValueError("invalid exp!")

        dir_name = f"model_save/pred_output/{RUN_NAME}/"
        file_name = os.path.join(dir_name, "test_results.json")
        if os.path.exists(file_name):
            with open(file_name, 'r') as reader:
                results = json.loads(reader.read())
            accu = round(100*results['predict_accuracy'], 2)
            f1 = round(100*results['predict_marco-f1'], 2)
            if metric == MACRO_F1:
                row_data.append(f1)
            elif metric == ACCU_F1:
                row_data.append(f"{accu} / {f1}")
            elif metric == CLASS_F1:
                pred_file_name = os.path.join(dir_name, "predict_results_None.txt")
                df = pd.read_csv(pred_file_name, sep='\t')
                y_pred = df['prediction'].tolist()
                if exp == BINARY:
                    true_file_name = f"data/{TEST}/dev_binary.json"
                    all_labels = ["supports", "refutes"]
                else:
                    true_file_name = f"data/{TEST}/dev.json"
                    all_labels = ["supports", "refutes", "not enough info"]
                y_true = []
                with open(true_file_name, 'r') as reader:
                    for line in reader.readlines():
                        item = json.loads(line)
                        y_true.append(item['label'])
                assert len(y_pred) == len(y_true)
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                class_f1_scores = [str(round(100*report[label]['f1-score'], 2)) for label in all_labels]
                class_f1_scores_str = '/'.join(class_f1_scores)
                row_data.append(f"{f1} ({class_f1_scores_str})")
                matrix = confusion_matrix(y_true, y_pred, labels=all_labels)
                df_matrix = pd.DataFrame(matrix, columns=all_labels)
                df_matrix['Actual'] = all_labels
                df_matrix = df_matrix.set_index('Actual')
                matrix_txt = df_matrix.to_string()
                report_txt = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
                report_writer.write(f"Train: {TRAIN}\tTest: {TEST}\nConfusion Matrix:\n")
                report_writer.write(matrix_txt)
                report_writer.write('\n\nClassification Report:\n')
                report_writer.write(report_txt)
                report_writer.write('\n'+'-'*50+'\n')
            else:
                raise ValueError("invalid metric!")
        else:
            row_data.append("N/A")
    all_results.append(row_data)


if highlight and metric == MACRO_F1 and len(all_results) > 2:
    # find biggest and second biggest for each column
    for j in range(1, len(all_results[0])):
        column_data = [all_results[i][j] for i in range(len(all_results))]
        max_idxs = heapq.nlargest(2, range(len(column_data)), key=column_data.__getitem__)
        all_results[max_idxs[0]][j] = "\\textbf{" + str(all_results[max_idxs[0]][j]) + "}"
        all_results[max_idxs[1]][j] = "\\underline{" + str(all_results[max_idxs[1]][j]) + "}"

df = pd.DataFrame(all_results, columns=columns)
csv_file = f"model_save/pred_output/results/{MODEL}_{exp}_{metric}_results.csv"
df.to_csv(csv_file, index=None)
print(csv_file)
report_writer.close()
