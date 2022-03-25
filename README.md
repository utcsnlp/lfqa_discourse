# lfqa_discourse
This repository contains the data and model for the following paper:

> [How Do We Answer Complex Questions: Discourse Structure of Long-form Answers](https://arxiv.org/abs/2203.11048) </br>
> Fangyuan Xu, Junyi Jessy Li, Eunsol Choi </br>
> ACL 2022
 

## Data

### Validity annotation

We release our validity annotations in `data/validity_annotation.jsonl`, which contains 1,591 (question, answer) pairs.

Each example is a json with the following field:
* `dataset`: The dataset this QA pair belongs to, one of [`NQ`, `ELI5`, `Web-GPT`]. Note that ELI5-model answer are distinguished with the `a_id` field mentioned below.
* `q_id`: The question id, same as the original NQ or ELI5 dataset.
* `a_id`: The answer id, same as the original ELI5 dataset. For NQ, we populate a dummy `a_id`. For machine generated answers, this field corresponds to the name of the model. For Web-GPT answers, this field will be 'Human demonstration'.
* `question`: The question.
* `answer_paragraph`: The answer paragraph.
* `answer_sentences`: The list of answer sentences, tokenized from the answer paragraph.
* `is_valid`: A boolean value indicating whether the qa pair is valid, values: [`True`, `False`].
* `invalid_reason`: A list of list, each list contains the invalid reason the annotator selected. The invalid reason is one of [`no_valid_answer`, `nonsensical_question`, `assumptions_rejected`, `multiple_questions`].

Here is a single validity annotation example: 

```
{
 'dataset': 'ELI5',
 'question': 'What do business people actually do all day?',
 'q_id': '1ep39o',
 'a_id': 'ca2m2in',
 'answer_paragraph': "Drink coffee. Stand around the watercooler. Show pictures of their kids to people who don't care. Dream of retirement.",
 'answer_sentences': ['Drink coffee.',
  'Stand around the watercooler.',
  "Show pictures of their kids to people who don't care.",
  'Dream of retirement.'],
  'is_valid': False,
  'invalid_reason': [['no_valid_answer', 'nonsensical_question'], 
  ['no_valid_answer'], ['no_valid_answer']],
 }
```


### Role annotation

We release our role annotations in `data/role_annotation.jsonl`, which contains 755 (question, answer) pairs with sentence-level role annotation.

Each example is a json with the following field:
* `dataset`: The dataset this QA pair belongs to, one of [`NQ`, `ELI5`, `Web-GPT`]. Note that ELI5-model answer are distinguished with the `a_id` field mentioned below.
* `q_id`: The question id, same as the original NQ or ELI5 dataset.
* `a_id`: The answer id, same as the original ELI5 dataset. For NQ, we populate a dummy `a_id` (1). For machine generated answers, this field corresponds to the name of the model. 
* `question`: The question.
* `answer_paragraph`: The answer paragraph.
* `answer_sentences`: The list of answer sentences, tokenized from the answer paragraph.
* `role_annotation`: The list of majority role (or adjudicated) role (if exists), for the sentences in `answer_sentences`. Each role is one of [`Answer`, `Answer - Example`, `Answer (Summary)`, `Auxiliary Information`, `Answer - Organizational sentence`, `Miscellaneous`]
* `raw_role_annotation`: A list of list, each list contains the raw role annotation for sentences in `answer_sentences`.

Here is a single role annotation example: 
```
{'dataset': 'ELI5',
 'q_id': '7c0lcl',
 'a_id': 'dpmf1xr',
 'question': 'Why are there such drastic differences in salaries between different countries?',
 'answer_paragraph': "I'm going to avoid discussing service industries, because they are drastically different and less subject to the global market (You can't work construction in Detroit and Munich on the same day)\n\nI'm mostly talking tech. \n\nThe biggest driver of disparity in tech jobs is cost of living. If it costs 2000 a month to live in Boston, and 200 a month to live in India, then salaries will reflect that. \n\nCompanies aren't in the business of lowering profits to give employees extra spending money.",
 'answer_sentences': ["I'm going to avoid discussing service industries, because they are drastically different and less subject to the global market (You can't work construction in Detroit and Munich on the same day)\n",
  "I'm mostly talking tech.",
  'The biggest driver of disparity in tech jobs is cost of living.',
  'If it costs 2000 a month to live in Boston, and 200 a month to live in India, then salaries will reflect that. ',
  "Companies aren't in the business of lowering profits to give employees extra spending money."],
 'role_annotation': ['Auxiliary Information', 'Miscellaneous', 'Answer (Summary)', 'Answer - Example', 'Auxiliary Information'],
 'raw_role_annotation': [['Miscellaneous',
   'Auxiliary Information',
   'Auxiliary Information'],
  ['Miscellaneous', 'Auxiliary Information', 'Answer'],
  ['Answer (Summary)', 'Answer (Summary)', 'Answer (Summary)'],
  ['Answer (Summary)', 'Answer - Example', 'Answer - Example'],
  ['Auxiliary Information', 'Auxiliary Information', 'Answer']]}
```

### NQ complex questions

We also release the 3,190 NQ questions with paragraph-level answer only that are identified as complex question requiring long-form answers in `data/nq_complex_qa.jsonl`. 

Each example is a json with the following field:
* `q_id`: The question id, same as the original NQ dataset.
* `question`: The question.
* `answer_paragraph`: The answer paragraph.
* `wiki_title`: The title of the wikipedia page the answer is extracted from.
* `split`: The split from the original NQ dataset, one of [`train`, `validation`]

Here is a single NQ complex example:

```
{
 'q_id': -8206924862280693153,
 'question': 'how does the word crucible related to the book crucible',
 'answer': " Miller originally called the play Those Familiar Spirits before renaming it as The Crucible . The word `` crucible '' is defined as a severe test or trial ; alternately , a container in which metals or other substances are subjected to high temperatures . The characters whose moral standards prevail in the face of death , such as John Proctor and Rebecca Nurse , symbolically refuse to sacrifice their principles or to falsely confess . ",
 'wiki_title': 'The Crucible',
 'split': 'validation'
}
```

## Model

We release our role classification model as well as the train/validation/test and out-of-domain data (NQ, WebGPT and ELI5-model) used in the paper, under the folder `role_classifier/data/`, with the following field: 

* `input_txt`: Input to T5 model.
* `target_txt`: Expected output with the role labels.
* `q_id`: The question id, same as those in `role_annotation.jsonl`.
* `a_id`: The answer id, same as those in `role_annotation.jsonl`.
* `target_all_labels`: A string containing all annotated roles, separated by comma for each sentence in the paragraph, used to calculate `Match-any` metric.

Please follow the instruction below to reproduce our result or run sentence classification on your own data.

### Install the requirements
Install conda and run the following command:
```bash
conda env create -f role_classifier/environment.yml
cd role_classifier
```

### Download the pre-trained model
Download the model from this [link](https://drive.google.com/file/d/1L_DbGhFqN-KBPJeTDFCAvX3RPZELJE9R/view?usp=sharing) and place it in the `model` directory.

### Generate input data 
To generate input data for role prediction, run the following script and pass in a csv file to ```-raw_data_path``` with two columns: ```question``` and ```answer```.
See `role_classifier/data/raw/sample_data.csv` for an example.
```bash
python generate_t5_input_csv_file.py \
--raw_data_path data/raw/sample_qa_data.csv \
--output_file_dir data/t5/
```

### Run predictions with pre-trained model 
Pass the generated file to ```test_file```, below is an example using the testing data. Sentence-level prediction will be saved to `<output_dir>/test_prediction.csv`

```bash
# t5 
python run_t5_role_classifier.py \
--train_file data/t5/train.csv \
--validation_file data/t5/validation.csv \
--test_file data/t5/test.csv \
--output_dir outputs/t5/test/ \
--do_predict \
--overwrite_output_dir \
--evaluation_strategy epoch \
--predict_with_generate \
--num_train_epoch 0 \
--model_name_or_path models/t5/
```


# Citation and contact
If you find our work helpful, please cite us as

```
@inproceedings{xu2022lfqadiscourse,
  title     = {How Do We Answer Complex Questions: Discourse Structure of Long-form Answers},
  author    = {Xu, Fangyuan and Li, Junyi Jessy and Choi, Eunsol},
  year      = 2022,
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics},
  note      = {Long paper}
}
```

Please contact Fangyuan Xu at `fangyuan[at]utexas.edu` if you have any questions or suggestions.
