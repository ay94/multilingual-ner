import os
import tempfile
import pytest
from multilingual_ner.evaluation import ReadNERData, align_dataset, check_labels


DUMMY_NER = """\
Ahmed B-PER
Younes I-PER
visited O
Cairo B-LOC
last O
week O
.

The O
UN B-ORG
met O
in O
Geneva B-LOC
.
"""


@pytest.fixture
def ner_file(tmp_path):
    p = tmp_path / "ner.txt"
    p.write_text(DUMMY_NER, encoding="utf-8")
    return str(p)


class TestReadNERData:
    def test_read_ner_file_sentence_count(self, ner_file):
        reader = ReadNERData()
        sentences, labels = reader.read_ner_file(ner_file, sentence_boundary=".")
        assert len(sentences) == 2
        assert len(labels) == 2

    def test_read_ner_file_tokens(self, ner_file):
        reader = ReadNERData()
        sentences, labels = reader.read_ner_file(ner_file, sentence_boundary=".")
        assert sentences[0][:6] == ["Ahmed", "Younes", "visited", "Cairo", "last", "week"]

    def test_read_ner_file_labels(self, ner_file):
        reader = ReadNERData()
        sentences, labels = reader.read_ner_file(ner_file, sentence_boundary=".")
        assert labels[0] == ["B-PER", "I-PER", "O", "B-LOC", "O", "O", "O"]

    def test_read_ner_file_second_sentence(self, ner_file):
        reader = ReadNERData()
        sentences, labels = reader.read_ner_file(ner_file, sentence_boundary=".")
        assert "UN" in sentences[1]
        assert "B-ORG" in labels[1]


class TestCheckLabels:
    def test_returns_set(self):
        labels = [["B-PER", "O", "B-LOC"], ["B-ORG", "O"]]
        result = check_labels(labels)
        assert isinstance(result, set)

    def test_contains_all_labels(self):
        labels = [["B-PER", "O", "B-LOC"], ["B-ORG", "O"]]
        result = check_labels(labels)
        assert result == {"B-PER", "O", "B-LOC", "B-ORG"}

    def test_empty_input(self):
        assert check_labels([]) == set()

    def test_single_sentence(self):
        labels = [["O", "B-PER", "I-PER"]]
        assert check_labels(labels) == {"O", "B-PER", "I-PER"}


class TestAlignDataset:
    def test_basic_alignment(self):
        labels = [["B-PERS", "I-PERS", "O", "B-LOC"]]
        mapping = {"B-PERS": "B-PER", "I-PERS": "I-PER"}
        result = align_dataset(labels, mapping)
        assert result[0] == ["B-PER", "I-PER", "O", "B-LOC"]

    def test_passthrough_unmapped(self):
        labels = [["B-LOC", "O"]]
        mapping = {"B-PER": "B-PER"}
        result = align_dataset(labels, mapping)
        assert result[0] == ["B-LOC", "O"]

    def test_map_to_O(self):
        labels = [["B-OTH", "I-OTH", "B-PER"]]
        mapping = {"B-OTH": "O", "I-OTH": "O"}
        result = align_dataset(labels, mapping)
        assert result[0] == ["O", "O", "B-PER"]

    def test_multiple_sentences(self):
        labels = [["B-PERS"], ["B-LOC", "O"]]
        mapping = {"B-PERS": "B-PER"}
        result = align_dataset(labels, mapping)
        assert result[0] == ["B-PER"]
        assert result[1] == ["B-LOC", "O"]

    def test_germeval_style_alignment(self):
        labels = [["B-LOCderiv", "I-LOCpart", "B-OTH", "B-PER"]]
        mapping = {
            "B-LOCderiv": "B-LOC", "I-LOCpart": "I-LOC",
            "B-OTH": "O",
        }
        result = align_dataset(labels, mapping)
        assert result[0] == ["B-LOC", "I-LOC", "O", "B-PER"]
