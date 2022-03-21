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
                          ,default='sample_input.csv')
    args = argparse.parse_args()

    data_df = pd.read_csv(args.raw_data_path)
    en_nlp = stanza.Pipeline('en', processors='tokenize')

    res_data = []
    for _, row in data_df.iterrows():
        if type(row['answer']) == str:
            input_line = [row['question']]
            target_line = []
            answer_paragraph = get_ans_sentence_with_stanza(row['answer'],
                                                            en_nlp)
            for idx, answer_sent in enumerate(answer_paragraph):
                sep_token = '[{}]'.format(idx)
                input_line.append(sep_token)
                input_line.append(answer_sent)
                target_line.append(sep_token)
                target_line.append('Answer') # note that this is a dummy label

            input_line = ' '.join(input_line)
            target_line = ' '.join(target_line)

            res_data.append({
                # first column is input
                'input_txt': input_line,
                # second column is target
                'target_txt': target_line,
            })
    # write output
    output_file_name = args.output_file_dir + args.output_file_name
    pd.DataFrame.from_records(res_data).to_csv(output_file_name, index=False)
    print(">> output {} data to {}".format(len(res_data),
                                           output_file_name))

if __name__ == "__main__":
    main()