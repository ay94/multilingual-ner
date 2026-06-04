"""
multilingual-ner
~~~~~~~~~~~~~~~~
Multilingual Named Entity Recognition toolkit.

Three components:
- evaluation: benchmark NER models against labelled datasets (seqeval + sklearn)
- extraction: run NER on project data at scale
- validation: Dash app for qualitative review of extraction outputs
"""

from .extraction import NamedEntityExtractions
from .evaluation import ReadNERData

__version__ = "0.1.0"
__all__ = ["NamedEntityExtractions", "ReadNERData"]
