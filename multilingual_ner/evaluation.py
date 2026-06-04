import re
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from transformers import pipeline
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import f1_score as seq_f1, precision_score as seq_precision, recall_score as seq_recall, \
    classification_report as seq_classification
from sklearn.metrics import f1_score as skl_f1, precision_score as skl_precision, recall_score as skl_recall, \
    classification_report as skl_classification


class ListDataset(Dataset):

    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


import re
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from transformers import pipeline
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import f1_score as seq_f1, precision_score as seq_precision, recall_score as seq_recall, \
    classification_report as seq_classification
from sklearn.metrics import f1_score as skl_f1, precision_score as skl_precision, recall_score as skl_recall, \
    classification_report as skl_classification
class ReadNERData:
    '''
        Read NER data class, it contains three different functionalities to read NER data.
        The NER data in the literature normally have consistent internal structure and flexible external structure.
        The internal structure is that it comes in word-label pair, this is consistent across all datasets.
        The external structure normally differ from dataset to another, which is divided to three main categories:
            - Data that comes in one text file, the read_ner_file function can be used in this case.
            - Data that comes in text files split into, train, val and test, this type you can either read individual file separately or put them all in one folder and read_ner_directory function.
            - Data that comes in directory where the directory contians various text files divide by topic (e.g, AQMAR), this type of data normally wikipedia articles that has been scraped and preprocessed into named entities structure.
        Most of the datasets fall under one of these types if your data is different you can add function to this class dedicated to your data.
    '''

    def __init__(self):
        pass

    def read_ner_file(self, file_path, sentence_boundary="."):
        '''
            Reads a single NER file from the specified file path.
            Processes the file line by line, extracting tokens and their corresponding labels.
            The function assumes that each line contains a token and its label separated by a space (this is standard internal strucutre).
            Blank lines are treated as sentence boundaries, sometimes they use . if so you can modify the sentence boundary argument.
            Returns a list of sentences (where each sentence is a list of tokens) and a corresponding list of label lists.

        '''
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines()]

        sentences = []
        labels = []
        current_sentence = []
        current_labels = []

        for line in lines:

            if line.split(" ")[0] == sentence_boundary:
                if current_sentence:
                    split = line.split()
                    if len(split) == 2:
                        token, label = split
                    current_sentence.append(token)
                    current_labels.append(label)

                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                try:
                    token, label = line.split()
                    current_sentence.append(token)
                    current_labels.append(label)
                except:
                    continue

        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)

        return sentences, labels



    def read_ner_directory(self, directory):
        '''
            Reads and processes all NER files in the specified directory.
            Use read_ner_file function to read each file
            Aggregates all sentences and labels from each file.
            Returns a combined list of all sentences and their corresponding label lists.
        '''
        all_sentences = []
        all_labels = []

        for file_name in tqdm(os.listdir(directory)):
            if file_name.endswith('.txt'):  # Assuming all NER files are .txt
                file_path = os.path.join(directory, file_name)
                sentences, labels = self.read_ner_file(file_path)
                all_sentences.extend(sentences)
                all_labels.extend(labels)

        return all_sentences, all_labels

    def read_dataset(self, dataset_name, ner_map, lang=None, split='test'):
        try:
            dataset = load_dataset(dataset_name)
        except:
            dataset = load_dataset(dataset_name, lang, trust_remote_code=True)
        ner_inv_map = {v: k for k, v in ner_map.items()}
        words_lists = []
        labels_lists = []
        try:
            self.data_split = dataset[split]
        except:
            self.data_split = dataset[f'{split}_{lang}']
        print(f'Generating {split} Split')
        for i in tqdm(range(len(self.data_split))):
            item = self.data_split.__getitem__(i)
            if 'tokens' in item:
                tokens = self.data_split.__getitem__(i)['tokens']
            elif 'words' in item:
                tokens = self.data_split.__getitem__(i)['words']

            if 'ner_tags' in item:
                labels = [ner_inv_map[tid] for tid in item['ner_tags']]
            elif 'tags' in item:
                labels = [ner_inv_map[tid] for tid in item['tags']]
            elif 'ner_tag' in item:
                labels = [ner_inv_map[tid] for tid in item['ner_tag']]
            elif 'ner' in item:
                labels = [ner_inv_map[tid] for tid in item['ner']]
            else:
                # Handle the case where none of the expected keys exist
                labels = []  # or some default value
            words_lists.append(tokens)
            labels_lists.append(labels)
        return words_lists, labels_lists

    def read_span_data(self, dataset_name, label_map, split='test'):
        dataset = load_dataset(dataset_name)
        self.data_split = dataset[split]
        example_list = []

        for i in tqdm(range(len(self.data_split))):
            example = {
                'text': self.data_split.__getitem__(i)['text'],
                'name_list': [entity['name'] for entity in self.data_split.__getitem__(i)['entities']],
                'span_list': [entity['span'] for entity in self.data_split.__getitem__(i)['entities']],
                'type_list': [label_map[entity['type']] for entity in self.data_split.__getitem__(i)['entities']]
            }
            example_list.append(example)

        return example_list

    def extract_spans(self):
        span_labels = []
        span_words = []
        span_data = []
        for i in tqdm(range(len(self.data_split))):
            try:
                spans = self.data_split.__getitem__(i)['spans']
                span_data.append(spans)
            except:
                break
            for span in spans:
                span_labels.append(span.split(':')[0].strip())
                span_words.append(span.split(':')[1].strip())
        return span_words, span_labels, span_data


def check_labels(label_list):
    return set([label for labels in label_list for label in labels])

def check_span_labels(label_list):
    return set([label for labels in label_list for label in labels['type_list']])

def align_dataset(label_lists, label_map):
    try:
        modified_label_lists = []
        for labels in label_lists:
            modified_labels = []
            for label in labels:
                if label in label_map:
                    modified_labels.append(label_map[label])
                else:
                    modified_labels.append(label)
            modified_label_lists.append(modified_labels)
    except:
        print('Data Already Aligned')
    return modified_label_lists


class NERDataset:

    def __init__(self, texts, tags, label_map, max_length, tokenizer):
        '''
       Constructor for the NERDataset class.
       Initializes the dataset with texts, tags, label mapping, maximum token length, and tokenizer name.
       Parameters:
          texts: A list of sentences, where each sentence is a list of words.
          tags: A corresponding list of tags for each word in the sentences.
          label_map: A dictionary mapping label names to label IDs.
          max_length: The maximum length of the token sequences.
          tokenizer: The name of the tokenizer to be used for tokenizing the sentences.
    '''

        self.texts = texts
        self.tags = tags
        self.label_map = label_map

        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Returns the number of sentences in the dataset.
        return len(self.texts)

    def __getitem__(self, item):
        '''
        Retrieves a single data item from the dataset at the specified index.
        This method processes the text and tags into a format suitable for model training or evaluation.
        Includes tokenization, padding, and conversion to tensor format.
        Returns a dictionary containing input IDs, attention masks, token type IDs, and labels, all as tensors.
    '''
        textlist = self.texts[item]
        tags = self.tags[item]
        tokens = []
        label_ids = []
        for word, label in zip(textlist, tags):

            word_tokens = self.tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Assigns the label ID to the first token of the word, and padding IDs to the remaining tokens.
                label_ids.extend([self.label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))

        # Truncates sequences if they exceed the maximum length, account for special tokens.
        special_tokens_count = self.tokenizer.num_special_tokens_to_add()
        if len(tokens) > self.max_length - special_tokens_count:
            tokens = tokens[: (self.max_length - special_tokens_count)]
            label_ids = label_ids[: (self.max_length - special_tokens_count)]

        # Add special tokens ([SEP] and [CLS]) and corresponding padding token label IDs.
        tokens += [self.tokenizer.sep_token]
        label_ids += [self.pad_token_label_id]
        token_type_ids = [0] * len(tokens)

        # Add the [CLS] TOKEN
        tokens = [self.tokenizer.cls_token] + tokens
        label_ids = [self.pad_token_label_id] + label_ids
        token_type_ids = [0] + token_type_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_length - len(input_ids)

        # Padding sequences to the max length.
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [self.pad_token_label_id] * padding_length

        # Ensure that all tensor dimensions are equal to the max_length.
        assert len(input_ids) == self.max_length
        assert len(attention_mask) == self.max_length
        assert len(token_type_ids) == self.max_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


class Evaluation:
    def __init__(self, label_ids, predictions, truth_map, pred_map, model_map) -> None:
        # Constructor for the Evaluation class.
        # Initializes the evaluation with true labels, predictions, and mapping dictionaries.
        # Parameters:
        #   label_ids: True label IDs for each token in the dataset.
        #   predictions: Predicted output from the model for each token.
        #   truth_map: A dictionary mapping label IDs to the dataset pre definde labels.
        #   pred_map: A dictionary mapping predicted label IDs to the model pre defined labels.
        #   model_map: An optional mapping for aligning model predictions with the standard annotation scheme.

        self.truth = label_ids
        self.preds = predictions
        self.truth_map = truth_map
        self.pred_map = pred_map
        self.model_map = model_map

    def create_classification_report(self, raw):
        # Converts a raw classification report string into a pandas DataFrame.
        # Parameters:
        #   raw: A raw string of the classification report.
        # Returns a DataFrame with columns for Tag, Precision, Recall, F1 score, and support.
        report = raw.strip().split('\n')
        lines = []
        for line in report[1:]:
            tokens = line.split()
            if line != '':
                if len(tokens) > 5:
                    del tokens[1]
                lines.append(tokens)
        return pd.DataFrame(lines, columns=['Tag', 'Precision', 'Recall', 'F1', 'support'])

    def convert_elements(self, element, model_map):
        # Define the dictionary

        # Check if the element is in the dictionary
        if element in model_map:
            return model_map[element]
        else:
            return element

    def align_predictions(self):
        # Aligns the model's predictions with the true labels.
        # Converts the logits to label IDs and maps them to the corresponding labels.
        # Ignores padding tokens in the true labels.
        # Returns two lists of lists: the true labels and the predicted labels, aligned at the token level.

        preds = np.argmax(self.preds, axis=2)
        batch_size, seq_len = preds.shape

        truth_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if self.truth[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                    truth_list[i].append(self.truth_map[self.truth[i][j]])
                    if self.model_map != None:
                        preds_list[i].append(self.convert_elements(self.pred_map[preds[i][j]], self.model_map))
                    else:
                        preds_list[i].append(self.pred_map[preds[i][j]])
        return truth_list, preds_list

    def compute_metrics(self):
        # Computes a variety of evaluation metrics based on the aligned predictions.
        # Uses both Seqeval and Sklearn metrics to evaluate the model performance.
        # Returns a dictionary containing precision, recall, F1 score, and detailed classification report
        # for both Seqeval and Sklearn evaluation methodologies.

        truth_list, preds_list = self.align_predictions()
        flat_truth_list = [item for sublist in truth_list for item in sublist]  # Flatten the lists.
        flat_preds_list = [item for sublist in preds_list for item in sublist]  # Flatten the lists.
        # Generate classification reports using Seqeval and Sklearn methods.
        seq_report = seq_classification(y_true=truth_list, y_pred=preds_list, digits=4)
        sk_report = skl_classification(y_true=flat_truth_list, y_pred=flat_preds_list, digits=4)

        return {
            'Seqeval':

                {"Precision": seq_precision(y_true=truth_list, y_pred=preds_list, average='micro'),
                 "Recall": seq_recall(y_true=truth_list, y_pred=preds_list, average='micro'),
                 "F1": seq_f1(y_true=truth_list, y_pred=preds_list, average='micro'),
                 "classification": self.create_classification_report(seq_report),
                 "output": {'y_true': truth_list, 'y_pred': preds_list}},

            'Sklearn':

                {"Precision": skl_precision(y_true=flat_truth_list, y_pred=flat_preds_list, average='macro'),
                 "Recall": skl_recall(y_true=flat_truth_list, y_pred=flat_preds_list, average='macro'),
                 "F1": skl_f1(y_true=flat_truth_list, y_pred=flat_preds_list, average='macro'),
                 "classification": self.create_classification_report(sk_report),
                 "output": {'y_true': flat_truth_list, 'y_pred': flat_preds_list}}
        }


class EvaluationOutput:
    def __init__(self, evaluation_output):
        self.evaluation_output = evaluation_output

    def get_classification(self, output_type='Seqeval'):
        return self.evaluation_output[output_type]['classification']

    def get_output(self, output_type='Seqeval'):
        return self.evaluation_output[output_type]['output']


class ModelEvaluation:
    '''
        The model evaluaiotn is divided into three steps:
            - Loading the model, this is done by get_model function.
            - Generating the evluation benchmark, this is done by generate_evaluation_data function.
            - Apply the model to the benchmark data and compute the performance, this is done by eval_fn.
        The output from this process is the classificaton report.
     '''

    # def __init__(self, model_name, words_lists, labels_lists, model_label_alignment=None,
    #              max_seq_length=512, batch_size=16, num_workers=2):
    def __init__(self, model_name, model_label_alignment=None,
                 max_seq_length=512, batch_size=16, num_workers=2):
        '''
            Constructor for the ModelEvaluation class.
            Initializes the model for Named Entity Recognition (NER) evaluation with the specified parameters.
            Parameters:
                model_name: The name of the pre-trained model to be used.
                words_lists: List of sentences, where each sentence is a list of words/tokens.
                labels_lists: Corresponding list of labels for each token in words_lists.
                model_label_alignment: Optional mapping of model labels to the standard annotation scheme.
                output_type: Format of the evaluation output ('Seqeval' by default) for the entity evaluation, sklearn for flat token evaluation.
                max_seq_length: Maximum sequence length for the model.
                batch_size: Batch size for processing.
                num_workers: Number of workers for data loading.
        '''
        self.model_name = model_name
        # self.words_lists = words_lists
        # self.labels_lists = labels_lists
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_label_alignment = model_label_alignment
        # self.dataset_label_map = {label: i for i, label in
        #                           enumerate(set(label for labels in labels_lists for label in labels))}
        # self.truth_map = {v: k for k, v in self.dataset_label_map.items()}
        # self.model, self.tokeniser, self.device = self.get_model()
        self.tokenizer, self.model, self.device = self.get_model()

    def get_model(self):
        '''
            Loads the specified pre-trained model and tokenizer.
            Determines if the model should run on GPU (CUDA) or CPU.
            Returns the model, tokenizer, and the device (GPU/CPU) it will run on.
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        try:
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        except:
            model = AutoModelForTokenClassification.from_pretrained(self.model_name, from_tf=True)
        model.to(device)
        # return model, model_tokenizer, device
        return model_tokenizer, model, device

    # def generate_evaluation_data(self):
    def generate_evaluation_data(self, words_lists, labels_lists):
        '''
            Prepares the evaluation dataset and dataloader.
            Converts words and labels into a format suitable for the model using a custom NERDataset class.
            Creates a DataLoader for efficient batch processing during model evaluation.
            Returns the dataset and dataloader objects.
        '''
        dataset = NERDataset(
            texts=[x for x in words_lists],
            tags=[x for x in labels_lists],
            label_map=self.dataset_label_map,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)
        return dataset, dataloader

    def eval_fn(self, model, data_loader, device, truth_map, pred_map, model_map=None):
        '''
            Evaluates the given model using the provided data loader.
            Parameters:
                model: The neural network model to be evaluated.
                data_loader: DataLoader object that provides batches of data for evaluation.
                device: The device (CPU or GPU) on which the model should be evaluated.
                truth_map: A mapping of true label IDs to their corresponding label names.
                pred_map: A mapping of predicted label IDs to their corresponding label names.
                model_map: Optional mapping for aligning model predictions with label IDs.
        '''
        model.eval()
        with torch.no_grad():
            preds = None
            labels = None
            # Iterating over the data loader.
            for data in tqdm(data_loader, total=len(data_loader)):
                for k, v in data.items():
                    data[k] = v.to(device)
                # Forward pass: compute the output of the model.
                outputs = model(**data)
                logits = outputs.logits  # Extract the raw model outputs (logits).
                truth = data['labels']  # Extract the true labels from the data.
                if logits is not None:
                    preds = logits if preds is None else torch.cat((preds, logits), dim=0)
                if truth is not None:
                    labels = truth if labels is None else torch.cat((labels, truth), dim=0)
            # Detaching predictions and labels from GPU and converting them to numpy arrays for evaluation.
            preds = preds.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            # Calculating evaluation metrics using the Evaluation class.
            evaluation = Evaluation(labels, preds, truth_map, pred_map, model_map)
            metrics = evaluation.compute_metrics()
        return metrics

    # def evaluate_model(self, output_type='Seqeval'):
    def evaluate_model(self, words_lists, labels_lists, output_type='Seqeval'):
        '''
            Evaluates the pre-trained model using the generated dataset and dataloader.
            Retrieves the model, tokenizer, and device (GPU/CPU).
            Generates the dataset and dataloader for evaluation.
            Perform the evaluation and returns the specified type of evaluation output, for example, entity evaluation if output_type is 'Seqeval'.
        '''
        # self.model, tokeniser, device = self.get_model()
        self.dataset_label_map = {label: i for i, label in
                                  enumerate(set(label for labels in labels_lists for label in labels))}
        self.truth_map = {v: k for k, v in self.dataset_label_map.items()}
        # dataset, dataloader = self.generate_evaluation_data()
        dataset, dataloader = self.generate_evaluation_data(words_lists, labels_lists)
        evaluation_output = self.eval_fn(self.model, dataloader, self.device, self.truth_map,
                                         self.model.config.id2label,
                                         self.model_label_alignment)
        return EvaluationOutput(evaluation_output)
        # return evaluation_output[output_type]['classification']


class ListDataset(Dataset):

    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]



class ProjectSpecificValidation:
    def __init__(self, model_name, project_data, text_col='text', model_label_alignment=None, batch_size=16,
                 max_length=512):
        # Initialize the instance with model name, validation sample, batch size, and max input length.
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.project_data = project_data
        self.text_col = text_col
        self.model_label_alignment = model_label_alignment
        self.tokenizer, self.ner_model = self.loading_models()

    def loading_models(self):
        print(f'Loading {self.model_name} Model')
        print()
        try:
            device = 0 if torch.cuda.is_available() else -1
            # Initialize the tokenizer with the specified model and maximum length.
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=self.max_length)
            # Initialize the Named Entity Recognition (NER) model pipeline.
            ner_model = pipeline("ner", model=self.model_name, tokenizer=tokenizer, device=device,
                                 aggregation_strategy="simple")
        except Exception as e:
            print(f"Model Loading Failed: An error occurred: {e}")
        return tokenizer, ner_model

    def extract_entity_outputs(self):
        entity_outputs = []
        try:
            # Convert the 'text' column of the validation sample to a list of sentences.
            sentences = self.project_data[self.text_col].tolist()
            # Run the NER model on the sentences.
            outputs = self.ner_model(ListDataset(sentences), batch_size=self.batch_size)
            # entity_outputs = ner_model(sentences, batch_size=self.batch_size)
            for output in tqdm(outputs):
                entity_outputs.append(output)

        except Exception as e:
            print(f"Extraction Faild: An error occurred: {e}")

        return entity_outputs

    def post_processing(self, entity_outputs):
        try:
            sentence_entities = []
            for entities in tqdm(entity_outputs):
                predicted_entities = []
                for i, entity in enumerate(entities):
                    current_word = entity['word']
                    current_start = entity['start']
                    current_end = entity['end']
                    current_entity_group = entity['entity_group']

                    if not predicted_entities:
                        predicted_entities.append((current_word, current_entity_group, current_start, current_end))
                    else:
                        last_word, last_tag, last_start, last_end = predicted_entities[-1]
                        if last_tag == current_entity_group and last_end == current_start:
                            # Merge with the previous word if they are part of the same entity and the current word starts right after the last one ends.
                            new_word = last_word + current_word
                            predicted_entities[-1] = (new_word, last_tag, last_start, current_end)
                        else:
                            predicted_entities.append((current_word, current_entity_group, current_start, current_end))

                # Remove index information for the final output if not needed
                sentence_entities.append(
                    [(re.sub('##', '', word), tag) for word, tag, start, end in predicted_entities])
        except Exception as e:
            print(f"Post Processing Faild: An error occurred: {e}")
        return sentence_entities

    def create_json_schema(self, extraction_outputs):
        try:
            json_schema = []
            # Iterate through the validation sample and extraction outputs.
            for index, row in tqdm(self.project_data.iterrows()):
                loc_entities, org_entities, misc_entities, pers_entities = [], [], [], []
                if self.model_label_alignment:
                    entities = [(entity[0], self.model_label_alignment[entity[1]]) for entity in
                                extraction_outputs[index]]
                else:
                    entities = extraction_outputs[index]
                # Classify and group entities by their type.
                for entity, type in entities:
                    if type == 'LOC':
                        loc_entities.append(entity)
                    elif type == 'ORG':
                        org_entities.append(entity)
                    elif type == 'MISC':
                        misc_entities.append(entity)
                    elif type == 'PERS':
                        pers_entities.append(entity)
                # Create a dictionary for each sentence with its associated information.
                sentence_dict = {
                    self.text_col: row[self.text_col],
                    "accountId": row['accountId'],
                    "message_id": row['message_id'],
                    "extractions": entities,
                    "LOC": loc_entities,
                    "ORG": org_entities,
                    "MISC": misc_entities,
                    "PERS": pers_entities
                }
                json_schema.append(sentence_dict)
        except Exception as e:
            print(f"Json Scheme  Faild: An error occurred: {e}")
        return json_schema

    def extract_outputs(self):
        # Run the NER extraction, post-processing, and create a JSON schema.
        print('Extract Entity Outputs')
        print()
        entity_outputs = self.extract_entity_outputs()
        print('Post Processing')
        print()
        post_processed_entities = self.post_processing(entity_outputs)
        print('Create Json Scheme')
        print()
        json_schema = self.create_json_schema(post_processed_entities)
        # Convert the JSON schema to a DataFrame.
        ner_output_df = pd.DataFrame(json_schema)
        return json_schema, ner_output_df, post_processed_entities, entity_outputs




def save_jsonl(file_name, data):
    # Open the file with the given file_name in write mode ('w').
    # This will create a new file or overwrite an existing file with the same name.
    with open(file_name, 'w') as file:
        # Iterate over each entry in the data list.
        for entry in data:
            # Convert the entry (a Python dictionary) into a JSON formatted string
            # and write it to the file.
            json.dump(entry, file)
            # Write a newline character after each JSON entry to separate the entries.
            file.write('\n')


def load_jsonl(file_path):
    # Initialize an empty list to store the loaded data.
    data = []
    # Open the file with the given file_path in read mode ('r').
    with open(file_path, 'r') as file:
        # Iterate over each line in the file.
        for line in file:
            # Strip whitespace from the ends of the line, which includes the newline character,
            # and convert the line from a JSON formatted string back into a Python dictionary.
            json_obj = json.loads(line.strip())
            # Append the dictionary to the data list.
            data.append(json_obj)
    # Return the list of dictionaries.
    return data


def check_tokenisation(tokenizer, data, text_col='text'):
    # Tokenize sentences and analyze truncation
    tokenized_data = []
    for index, row in tqdm(data.iterrows()):
        tokens = tokenizer.tokenize(row[text_col])
        tokenized_length = len(tokens)
        truncated = tokenized_length > tokenizer.model_max_length
        truncation_amount = max(0, tokenized_length - tokenizer.model_max_length)

        tokenized_data.append({
            "message_id": row['message_id'],
            "text": row[text_col],
            "num_tokens": tokenized_length,
            "truncated": truncated,
            "truncation_amount": truncation_amount
        })

    # Convert to DataFrame for better visualization
    tokenized_df = pd.DataFrame(tokenized_data)
    return tokenized_df


def generate_sample_file(sample_df, text_col='text'):
    sample_df['Mistakes (Precision)'] = pd.Series(dtype='int')
    sample_df['Missing (Recall)'] = pd.Series(dtype='int')
    sample_df['Google Translation'] = ''
    sample_df['Tokenisation Issue'] = ''
    sample_df['Notes'] = ''
    columns_order = ['Mistakes (Precision)', 'Missing (Recall)', 'Google Translation', 'extractions', text_col, 'LOC',
                     'ORG', 'MISC', 'PERS', 'Tokenisation Issue', 'Notes', 'accountId', 'message_id',
                     'num_tokens', 'truncated']
    return sample_df[columns_order]


def extract_results(fh, exclusions):
    TAGs = ['LOC', 'PER', 'ORG']
    for TAG in TAGs:
        entity_results = []
        directory = Path(fh.cr_fn('outputs'))
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                # Construct the file's full path
                file_path = os.path.join(subdir, file)
                excluded_folder = file_path.split('/')[-2]
                if excluded_folder != 'wikiann-span':
                    # Open and read the file
                    file_name = file_path.split('/')[-1]
                    model_name = file_path.split('/')[-2]
                    if 'seqeval' in file_name:
                        df = pd.read_csv(file_path)
                        df['Model Name'] = model_name
                        df['Data Name'] = file_name.split('-')[0]
                        print()
                        print(df[df['Tag'] == TAG])
                        entity_results.append(df[df['Tag'] == TAG])
        pd.concat(entity_results).to_csv(
            fh.cr_fn(f'outputs/consolidated-{TAG}.csv'),
            index=False
        )



class SpanEvaluation:
    def __init__(self, model_name, model_label_alignment=None,
                 max_seq_length=512, batch_size=16, num_workers=2):

        self.model_name = model_name

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_label_alignment = model_label_alignment

        self.tokenizer, self.ner_model, self.device = self.get_model()

    def get_model(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        ner_model = pipeline("ner", model=self.model_name, tokenizer=tokenizer, device=device,
                             aggregation_strategy="simple")

        return tokenizer, ner_model, device

    def align_model_output(self, model_outputs):
        output_list = []
        for output in tqdm(model_outputs, 'Aling Model Outputs'):
            names = []
            spans = []
            types = []
            for entity in output:
                names.append(entity['word'])
                spans.append([entity['start'], entity['end']])
                types.append(entity['entity_group'])
            output = {
                'name_list': names,
                'span_list': spans,
                'type_list': types
            }
            output_list.append(output)

        return output_list

    def generate_evaluation_data(self, span_dataset):
        model_outputs = []
        pipeline_outputs = self.ner_model(ListDataset([data['text'] for data in span_dataset]))
        for output in tqdm(pipeline_outputs, 'Extract Pipeline Outputs'):
            model_outputs.append(output)
        predictions = self.align_model_output(model_outputs)
        return model_outputs, predictions

    def evaluate_model(self, predictions, benchmark):
        type_mismatches = 0
        span_errors = 0
        extra_predictions = 0
        matched_benchmark_indices = set()

        for pred_entity in predictions:
            closest_match = None
            closest_distance = float('inf')
            closest_index = None

            # Find the closest match in the benchmark
            for index, bench_entity in enumerate(benchmark):
                distance = abs(pred_entity['span'][0] - bench_entity['span'][0]) + abs(
                    pred_entity['span'][1] - bench_entity['span'][1])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_match = bench_entity
                    closest_index = index

            # Evaluate type and span accuracy
            if closest_match:
                if pred_entity['type'] != closest_match['type']:
                    type_mismatches += 1
                if pred_entity['span'] != closest_match['span']:
                    span_errors += 1
                matched_benchmark_indices.add(closest_index)

        # Count missed entities (not matched by any prediction)
        missed_entities = len(benchmark) - len(matched_benchmark_indices)

        return {
            'type_mismatches': type_mismatches,
            'span_errors': span_errors,
            'extra_predictions': len(predictions) - len(matched_benchmark_indices),
            'missed_entities': missed_entities
        }

    def evaluation_summary(self, evaluation_output):
        # Initialize counters for summary statistics
        total_type_mismatches = total_span_errors = total_extra_predictions = total_missed_entities = 0

        # Sum up the counts from each evaluation
        for result in evaluation_output:
            total_type_mismatches += result['type_mismatches']
            total_span_errors += result['span_errors']
            total_extra_predictions += result['extra_predictions']
            total_missed_entities += result['missed_entities']

        # Calculate the number of evaluations
        num_evaluations = len(evaluation_output)

        # Calculate averages
        avg_type_mismatches = total_type_mismatches / num_evaluations
        avg_span_errors = total_span_errors / num_evaluations
        avg_extra_predictions = total_extra_predictions / num_evaluations
        avg_missed_entities = total_missed_entities / num_evaluations

        # Print summary statistics
        summary_stats = {
            'Total Type Mismatches': total_type_mismatches,
            'Average Type Mismatches': avg_type_mismatches,
            'Total Span Errors': total_span_errors,
            'Average Span Errors': avg_span_errors,
            'Total Extra Predictions': total_extra_predictions,
            'Average Extra Predictions': avg_extra_predictions,
            'Total Missed Entities': total_missed_entities,
            'Average Missed Entities': avg_missed_entities
        }
        return pd.DataFrame(summary_stats, index=['Stats'])

    def evaluate(self, span_dataset):
        self.model_outputs, self.predictions = self.generate_evaluation_data(span_dataset)

        self.evaluation_output = []
        for benchmark, prediction in tqdm(zip(span_dataset, self.predictions), 'Evaluation'):
            benchmark_entities = [{'name': name, 'span': span, 'type': type_} for name, span, type_ in
                                  zip(benchmark['name_list'], benchmark['span_list'], benchmark['type_list'])]

            predicted_entities = [{'name': name, 'span': span, 'type': type_} for name, span, type_ in
                                  zip(prediction['name_list'], prediction['span_list'], prediction['type_list'])]

            # Apply the evaluation function
            self.evaluation_output.append(self.evaluate_model(predicted_entities, benchmark_entities))

        return self.evaluation_summary(self.evaluation_output)











