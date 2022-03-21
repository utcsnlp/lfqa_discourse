# lfqa_discourse
A repository for ACL 2022 paper "How Do We Answer Complex Questions: Discourse Structure of Long-form Answers".

## Data

We release our discourse annotation in `data/discourse_data.jsonl`, which contains 1,592 (question, answer) pairs with validity annotation and 755 (question, answer) pairs with sentence-level role annotation.

Each example is a json with the following field:
* `type`: The type of the annotation, one of [`validity`,`role`].
* `dataset`: The dataset this QA pair belongs to, one of [`NQ`, `ELI5`, `WebGPT`]. Note that ELI5-model answer are distinguished with the `a_id` field mentioned below.
* `q_id`: The question id, same as the original NQ or ELI5 dataset.
* `a_id`: The answer id, same as the original ELI5 dataset. For NQ, we populate a dummy `a_id` (1). For machine generated answers, this field corresponds to the name of the model. 
* `question`: The question.
* `answer_paragraph`: The answer paragraph.
* `answer_sentences`: The list of answer sentences, tokenzied from the answer paragraph.
* `is_valid`: (validity data only) A boolean value indicating whether the qa pair is valid.
* `question_validity_count`: (validity data only) The number of annotator that selected this pair as valid.
* `invalid_reason`: (validity data only) A string consisting comma-separated invalid reason annotations, in the format of `{$invalid_reason: count of annotations for $invalid_reason}`.
* `role_annotation`: (role data only) The list of majority role (or adjudicated) role (if exists), for the sentence in `answer_sentences`.
* `detail_role_annotation`: (role data only) The list of strings with comma-separated detailed role annotations for the sentence in `answer_sentences`, in the format of `{role: count of annotation for role}`.


We also release the 3,190 NQ questions with paragraph-level answer only that are identified as complex question requiring long-form answers in `data/nq_non_factoid_qa.jsonl`. 

Each example is a json with the following field:
* `q_id`: The question id, same as the original NQ dataset.
* `question`: The question.
* `answer_paragraph`: The answer paragraph.
* `wiki_title`: The title of the wikipedia page the answer is extracted from.
* `split`: The split from the original NQ dataset, one of [`train`, `validation`]

## Model

We release our role classification model as well as the train/validation/test and out-of-domain data (NQ, WebGPT and ELI5-model) used in the paper.Please follow the instruction below to reproduce our result or run sentence classification on your own data.

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