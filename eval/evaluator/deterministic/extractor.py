# download model (bash) code: python -m spacy download en_core_web_sm
import spacy

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed.\n"
                "Fix this by running:\n"
                "  python -m spacy download en_core_web_sm\n"
                "or adding it to your Dockerfile."
            )
    return _nlp

class FeatureExtractor:
    def __init__(self):
        pass # avoid downloading spacy 

    def extract(self, text: str):
        nlp = get_nlp()
        doc = nlp(text)

        #tokens = {t.text.lower() for t in doc if not t.is_stop}
        entities = {chunk.text.lower() for chunk in doc.noun_chunks}

        return {
            #"tokens": tokens,
            "entities": entities,
            #"doc": doc
        }