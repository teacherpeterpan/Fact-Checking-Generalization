import json
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
from collections import Counter

datasets = ['FEVER-paragraph', 'FEVER-sentence', 'VitaminC', 'Climate-FEVER-paragraph', 'Climate-FEVER-sentence', 'Sci-Fact-paragraph', 'Sci-Fact-sentence', 'PUBHEALTH', 'COVID-Fact', 'FAVIQ', 'FoolMeTwice']
columns = ['Dataset', "# (%) Supports", "# (%) Refutes", "# (%) Not enough info", "# (%) Supports", "# (%) Refutes", "# (%) Not enough info", "# Total", "# avg. claim tokens", "# avg. evidence tokens"]
stat = []

for dataset in datasets:
    print(dataset)
    claim_tokens_cnt = 0
    evidence_tokens_cnt = 0
    total_cnt = 0
    stat_row = [dataset]
    for split in ["train", "dev"]:
        all_labels = []
        with open(f"./{dataset}/{split}.json", 'r') as reader:
            for line in tqdm(reader.readlines()):
                item = json.loads(line.strip())
                all_labels.append(item['label'])
                claim_tokens_cnt += len(word_tokenize(item['claim']))
                evidence_tokens_cnt += len(word_tokenize(item['evidence']))
        total_cnt += len(all_labels)
        counter = Counter(all_labels)
        for label in ["supports", "refutes", "not enough info"]:
            label_cnt = counter.get(label, 0)
            label_percent = round(100 * label_cnt / len(all_labels), 1)
            stat_row.append(f"{label_cnt:,d} ({label_percent})")
    stat_row.append(f"{total_cnt:,d}")
    avg_claim_tokens = round(claim_tokens_cnt / total_cnt, 1)
    avg_evidence_tokens = round(evidence_tokens_cnt / total_cnt, 1)
    stat_row.append(avg_claim_tokens)
    stat_row.append(avg_evidence_tokens)
    stat.append(stat_row)

df = pd.DataFrame(stat, columns=columns)
df.to_csv("dataset_statistics.csv", index=None)
