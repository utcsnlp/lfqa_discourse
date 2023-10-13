"""To run:
 python _run_t5_role_classifier.py \
 --input_data_csv data/t5/input_alpaca.csv \
 --output_json_file_path input_alpaca.json

 input_data_csv should contain columns [question, answer_sentences]
 output_json_file will contain [predicted_labels]
"""


from argparse import ArgumentParser
import pandas as pd
from datetime import date
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import ast
import re
from tqdm import tqdm

model = AutoModelForSeq2SeqLM.from_pretrained("fangyuan/lfqa_role_classification").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("fangyuan/lfqa_role_classification")

def process_t5_output(input_txt, output_txt):
    pred_roles = []
    answer_sentence = re.split('\[\d+\] ', input_txt)
    answer_sentence = answer_sentence[1:]
    sentence_idx = re.findall('\[\d+\]', input_txt)
    idx_to_sentence = zip(sentence_idx, answer_sentence)
    pred_role = re.split('\[\d+\] ', output_txt)[1:]
    pred_idx = re.findall('\[\d+\]', output_txt)
    idx_to_role = {
        idx: role.strip() for (idx, role) in zip(pred_idx, pred_role)
    }
    print(input_txt, output_txt)
    for _, (idx, sentence) in enumerate(idx_to_sentence):
        pred_role = ' ' if idx not in idx_to_role else idx_to_role[idx]
        mapped_pred_role = role_mappings[pred_role]
        pred_roles.append(mapped_pred_role)
    return output_txt, pred_roles

role_mappings = {
    'Answer': 'Answer',
    'Answer (Summary)': 'Summary',
    'Auxiliary Information': 'Auxiliary Information',
    'Answer - Example': 'Example',
    'Miscellaneous': 'Miscellaneous',
    'Answer - Organizational sentence': 'Organizational sentence',
    ' ': ' ',
}


def predict(input_txt):
    input_ids = tokenizer(input_txt, return_tensors='pt', max_length=500, truncation=True).input_ids.to('cuda')
    # limit the input txt to be less than 500 tokens
    outputs = model.generate(input_ids, max_length=512)
    output_txt = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return process_t5_output(input_txt, output_txt)


def main():
    argparse = ArgumentParser()
    argparse.add_argument("--input_json_file_path", dest='input_json_file_path', required=True)
    argparse.add_argument("--output_json_file_path", dest='output_json_file_path', required=True)
    args = argparse.parse_args()

    data_df = pd.read_json(args.input_json_file_path)
    predicted_labels = []
    input_txts, pred_txts = [], []
    for i, data in tqdm(data_df.iterrows(), total=len(data_df)):
        # generate input text
        input_line = [data['question']]
        answer_paragraph = data['answer']
        for idx, answer_sent in enumerate(answer_paragraph):
            sep_token = '[{}]'.format(idx)
            input_line.append(sep_token)
            input_line.append(answer_sent)
        input_line = ' '.join(input_line)
        input_txts.append(input_line)
        pred_txt, pred_labels = predict(input_line)
        predicted_labels.append(pred_labels)
        pred_txts.append(pred_txt)

    data_df['predicted_labels'] = predicted_labels
    data_df['pred_txt'] = pred_txts
    data_df['input_txt'] = input_txts

    data_df.to_json(args.output_json_file_path, orient='records')



if __name__ == "__main__":
    main()