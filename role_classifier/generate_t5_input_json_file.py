"""generate dataset files for t5 model to predict roles.
 Usage:
    python3 generate_t5_input_csv_file.py
    --raw_data_path=<raw_data_path>
    --output_file_dir=<output_file_dir>
 """

from argparse import ArgumentParser
import pandas as pd
from datetime import date
import os
import ast
import stanza
stanza.download('en', processors='tokenize')


def get_ans_sentence_with_stanza(answer_paragraph, pipeline,
                                 is_offset=False):
    '''sentence segmentation with stanza'''
    answer_paragraph_processed = pipeline(answer_paragraph)
    sentences = []
    for sent in answer_paragraph_processed.sentences:
        if is_offset:
            sentences.append((sent.tokens[0].start_char, sent.tokens[-1].end_char))
        else:
            sentence = answer_paragraph[sent.tokens[0].start_char:sent.tokens[-1].end_char + 1]
            sentences.append(sentence.strip())
    return sentences

def main():
    argparse = ArgumentParser()
    argparse.add_argument("--raw_data_path", dest='raw_data_path', default='data/raw/sample_qa_data.csv')
    argparse.add_argument("--output_file_dir", dest='output_file_dir'
                          ,default='data/t5/')
    argparse.add_argument("--output_file_name", dest='output_file_name'
                          ,default='sample_input.json')
    args = argparse.parse_args()

    data_df = pd.read_csv(args.raw_data_path)
    en_nlp = stanza.Pipeline('en', processors='tokenize')

    res_data = []
    for _, row in data_df.iterrows():
        if type(row['answer']) == str:
            answer_paragraph = get_ans_sentence_with_stanza(row['answer'],
                                                            en_nlp)

            res_data.append({
                'question': row['question'],
                'answer': [sent for idx, sent in enumerate(answer_paragraph)],
            })
    # write output
    output_file_name = args.output_file_dir + args.output_file_name
    pd.DataFrame.from_records(res_data).to_json(output_file_name, orient='records')
    print(">> output {} data to {}".format(len(res_data),
                                           output_file_name))

if __name__ == "__main__":
    main()