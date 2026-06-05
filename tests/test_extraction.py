import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from multilingual_ner.extraction import NamedEntityExtractions


SAMPLE_DF = pd.DataFrame({
    "text": [
        "Ahmed visited Cairo last week.",
        "The UN met in Geneva.",
        "Apple released a product in Cupertino.",
    ],
    "message_id": ["msg_1", "msg_2", "msg_3"],
    "accountId":  ["acc_1", "acc_1", "acc_2"],
})

RAW_OUTPUT = [
    [{"word": "Ahmed",    "entity_group": "PER",  "start": 0,  "end": 5}],
    [{"word": "UN",       "entity_group": "ORG",  "start": 4,  "end": 6},
     {"word": "Geneva",   "entity_group": "LOC",  "start": 14, "end": 20}],
    [{"word": "Apple",    "entity_group": "ORG",  "start": 0,  "end": 5},
     {"word": "Cupertino","entity_group": "LOC",  "start": 27, "end": 36}],
]


@pytest.fixture
def extractor():
    with patch.object(NamedEntityExtractions, "loading_models") as mock_load:
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 512
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = iter(RAW_OUTPUT)
        mock_load.return_value = (mock_tokenizer, mock_pipeline)
        obj = NamedEntityExtractions.__new__(NamedEntityExtractions)
        obj.model_name = "mock-model"
        obj.batch_size = 8
        obj.max_length = 512
        obj.project_data = SAMPLE_DF
        obj.text_col = "text"
        obj.model_label_alignment = None
        obj.tokenizer, obj.ner_model = mock_tokenizer, mock_pipeline
        yield obj


class TestPostProcessing:
    def test_basic_entities(self, extractor):
        result = extractor.post_processing(RAW_OUTPUT)
        assert len(result) == 3

    def test_per_entity_extracted(self, extractor):
        result = extractor.post_processing(RAW_OUTPUT)
        entities_0 = [(w, t) for w, t, *_ in result[0]]
        assert ("Ahmed", "PER") in entities_0

    def test_two_entities_second_sentence(self, extractor):
        result = extractor.post_processing(RAW_OUTPUT)
        assert len(result[1]) == 2

    def test_subword_hash_stripped(self, extractor):
        raw = [[{"word": "##med", "entity_group": "PER", "start": 2, "end": 5}]]
        result = extractor.post_processing(raw)
        word = result[0][0][0]
        assert "##" not in word


class TestCreateJsonSchema:
    def test_schema_length(self, extractor):
        post = extractor.post_processing(RAW_OUTPUT)
        schema = extractor.create_json_schema(post)
        assert len(schema) == 3

    def test_text_field_preserved(self, extractor):
        post = extractor.post_processing(RAW_OUTPUT)
        schema = extractor.create_json_schema(post)
        assert schema[0]["text"] == "Ahmed visited Cairo last week."

    def test_per_field(self, extractor):
        post = extractor.post_processing(RAW_OUTPUT)
        schema = extractor.create_json_schema(post)
        assert "Ahmed" in schema[0]["PER"]

    def test_loc_field(self, extractor):
        post = extractor.post_processing(RAW_OUTPUT)
        schema = extractor.create_json_schema(post)
        assert "Geneva" in schema[1]["LOC"]

    def test_org_field(self, extractor):
        post = extractor.post_processing(RAW_OUTPUT)
        schema = extractor.create_json_schema(post)
        assert "UN" in schema[1]["ORG"]

    def test_message_id_present(self, extractor):
        post = extractor.post_processing(RAW_OUTPUT)
        schema = extractor.create_json_schema(post)
        assert schema[0]["message_id"] == "msg_1"
