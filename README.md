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
* `answer_sentences`: The list of answer sentences, tokenzied from the answer paragraph.
* `is_valid`: A boolean value indicating whether the qa pair is valid.
* `question_validity_count`: The number of annotator that selected this pair as valid.
* `invalid_reason`: A string consisting comma-separated invalid reason annotations, in the format of `{$invalid_reason: count of annotations for $invalid_reason}`.

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
  'invalid_reason': 'no_valid_answer: 3,nonsensical_question: 1',
  'question_validity_count': 0
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
* `answer_sentences`: The list of answer sentences, tokenzied from the answer paragraph.
* `role_annotation`: The list of majority role (or adjudicated) role (if exists), for the sentence in `answer_sentences`.
* `detail_role_annotation`: The list of strings with comma-separated detailed role annotations for the sentence in `answer_sentences`, in the format of `{role: count of annotation for role}`.

Here is a single role annotation example: 
```
{
 'dataset': 'ELI5',
 'q_id': '29gy6c',
 'a_id': 'ciksl73',
 'question': 'How does a silencer on a fire arm work?',
 'answer_paragraph': "The aim of a silencer is to break up/soften the noise of the weapon firing. It does this by directing the air leaving the muzzle through a series of baffles, slowing and redirecting the air so that it will form a 'softer' noise, rather than a single loud pulse.\n\nThe noise you hear in the movies is not representative of the average silenced weapon, but is a plot device to let bad (or good) guys do their job steathily. In practice, the silencer will reduce the noise and make it harder to pinpoint, but will not give anything like as significant a reduction in volume",
 'answer_sentences': ['The aim of a silencer is to break up/soften the noise of the weapon firing.',
  "It does this by directing the air leaving the muzzle through a series of baffles, slowing and redirecting the air so that it will form a 'softer' noise, rather than a single loud pulse.\n",
  'The noise you hear in the movies is not representative of the average silenced weapon, but is a plot device to let bad (or good) guys do their job steathily. ',
  'In practice, the silencer will reduce the noise and make it harder to pinpoint, but will not give anything like as significant a reduction in volume'],
 'role_annotation': ['Answer',
  'Answer (Summary)',
  'Auxiliary Information',
  'Auxiliary Information'],
 'detail_role_annotation': ['Answer: 2,Answer (Summary): 1',
  'Answer: 1,Answer (Summary): 2',
  'Auxiliary Information: 3',
  'Answer: 1,Auxiliary Information: 2']
}
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


# Citations
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

# Contact
Please contact at `fangyuan@utexas.edu` if you have any questions.
