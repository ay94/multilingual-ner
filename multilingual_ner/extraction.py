import re
import json
import torch

import pandas as pd
from tqdm.notebook import tqdm
from transformers import pipeline
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ListDataset(Dataset):

    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


class NamedEntityExtractions:
    def __init__(self, model_name, project_data, text_col='text', TokenizerLoader=None, tokenizer_name=None,
                 model_label_alignment=None, batch_size=16,
                 max_length=512):
        # Initialize the instance with model name, validation sample, batch size, and max input length.
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.project_data = project_data
        self.text_col = text_col
        self.model_label_alignment = model_label_alignment
        self.tokenizer, self.ner_model = self.loading_models(TokenizerLoader, tokenizer_name)

    def loading_models(self, TokenizerLoader, tokenizer_name):
        print(f'Loading {self.model_name} Model')
        print()
        # try:
        device = 0 if torch.cuda.is_available() else -1
        # Initialize the tokenizer with the specified model and maximum length.
        if TokenizerLoader and tokenizer_name:
            print(f'I am {TokenizerLoader} and {tokenizer_name}')
            tokenizer = TokenizerLoader.from_pretrained(tokenizer_name, model_max_length=self.max_length)

        elif TokenizerLoader:
            print(f'I am {TokenizerLoader} and {tokenizer_name}')
            tokenizer = TokenizerLoader.from_pretrained(self.model_name, model_max_length=self.max_length)
        else:
            print(f'I am {TokenizerLoader} and {tokenizer_name}')
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=self.max_length)
        # Initialize the Named Entity Recognition (NER) model pipeline.
        ner_model = pipeline("ner", model=self.model_name, tokenizer=tokenizer, device=device,
                             aggregation_strategy="simple")
        # except Exception as e:
        # print(f"Model Loading Failed: An error occurred: {e}")
        return tokenizer, ner_model

    def map_entities(self, element, model_map):
        # Define the dictionary

        # Check if the element is in the dictionary
        if element in model_map:
            return model_map[element]
        else:
            return element

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
                try:
                    sentence_entities.append(
                        [(re.sub('##', '', word), self.map_entities(tag, self.model_label_alignment), start, end) for
                         word, tag, start, end in predicted_entities])
                except:
                    sentence_entities.append(
                        [(re.sub('##', '', word), tag, start, end) for word, tag, start, end in predicted_entities])
        except Exception as e:
            print(f"Post Processing Faild: An error occurred: {e}")
        return sentence_entities

    def create_json_schema(self, extraction_outputs):
        try:
            json_schema = []
            # Iterate through the validation sample and extraction outputs.
            for index, row in tqdm(self.project_data.iterrows()):
                loc_entities, org_entities, misc_entities, per_entities = [], [], [], []
                # if self.model_label_alignment:
                #     entities = [(entity[0], self.map_entities(entity[1], self.model_label_alignment)) for entity in
                #         extraction_outputs[index]]
                # else:
                entities = extraction_outputs[index]
                # Classify and group entities by their type.
                for entity, type, _, _ in entities:
                    if type == 'LOC':
                        loc_entities.append(entity)
                    elif type == 'ORG':
                        org_entities.append(entity)
                    elif type == 'MISC':
                        misc_entities.append(entity)
                    elif type == 'PER':
                        per_entities.append(entity)
                # Create a dictionary for each sentence with its associated information.
                sentence_dict = {
                    self.text_col: row[self.text_col],
                    "accountId": row['accountId'],
                    "message_id": row['message_id'],
                    "extractions": entities,
                    "LOC": loc_entities,
                    "ORG": org_entities,
                    "MISC": misc_entities,
                    "PER": per_entities
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
