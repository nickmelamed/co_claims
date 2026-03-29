# download model (bash) code: python -m spacy download en_core_web_sm
import spacy

class FeatureExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str):
        doc = self.nlp(text)

        tokens = {t.text.lower() for t in doc if not t.is_stop}
        entities = {chunk.text.lower() for chunk in doc.noun_chunks}

        return {
            "tokens": tokens,
            "entities": entities,
            "doc": doc
        }