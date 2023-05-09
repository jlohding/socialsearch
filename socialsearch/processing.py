import re
import spacy
from spacy.tokenizer import _get_regex_pattern
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CaptionProcessingTransformer(BaseEstimator, TransformerMixin):
    # combines CaptionFeatureEngineeringTransformer and CaptionEncoderTransformer
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = CaptionFeatureEngineeringTransformer().transform(X,y)
        embeddings = CaptionEncoderTransformer().transform(X,y)
        return np.column_stack([embeddings, features])

class CaptionFeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    # sklearn.pipeline compatible
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # feature engineering new variables
        num_tagged_users = X["tagged_users"].apply(len)
        caption_length = X["caption"].apply(len)
        num_caption_hashtags = X["caption_hashtags"].apply(len)
        num_caption_mentions = X["caption_mentions"].apply(len)
        tag_in_caption = X["caption"].apply(lambda text: 1 if "@" in text else 0)

        return np.column_stack([num_tagged_users, caption_length, num_caption_hashtags, num_caption_mentions, tag_in_caption])

class CaptionEncoderTransformer(BaseEstimator, TransformerMixin):
    # sklearn.pipeline compatible
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        processor = CaptionEncoder()
        embeddings = processor.preprocess_encode(X["caption"])
        return embeddings

class CaptionEncoder:
    def __init__(self, spacy_model="en_core_web_sm", sbert_model="all-MiniLM-L6-v2"):
        self.nlp = spacy.load(spacy_model) # spacy preprocessor
        self.encoder = SentenceTransformer(sbert_model) # embedding model
        self.__setup()
    
    def __setup(self) -> None:
        '''Setup spacy preprocessor'''
        re_token_match = _get_regex_pattern(self.nlp.Defaults.token_match)
        re_token_match = f"({re_token_match}|#\w+|\w+-\w+|@\w+)"
        self.nlp.tokenizer.token_match = re.compile(re_token_match).match
    
    def __normalise(self, sentence: str) -> str:
        '''Normalise and preprocess one sentence of text data'''
        doc = self.nlp(sentence)
        normalised_text = []
        for token in doc:
            if not token.is_punct and not token.is_stop and not token.is_space:
                if token.lemma_ == token.text.upper():
                    normalised_text.append(token.lemma_)
                else:
                    normalised_text.append(token.lemma_.lower())

        return " ".join(normalised_text)        
        
    def preprocess(self, sentences: 'iterable[str]') -> list[str]:
        '''Run preprocessor on list of input strings'''
        sentences = ["" if s is None else s for s in sentences]
        normalised_sentences = [self.__normalise(s) for s in sentences]
        return normalised_sentences
    
    def encode(self, normalised_sentences: list[str]) -> np.array:
        '''Runs SentenceTransformer encoder on preprocessed language data
        
        Args: 
            normalised_sentences: List of preprocessed text data
        
        Returns:
            Numeric vector representation in a numpy array 
        '''
        embeddings = self.encoder.encode(normalised_sentences)
        return embeddings
    
    def preprocess_encode(self, sentences: 'iterable[str]') -> np.array:
        '''Curried function combining preprocess and encode methods'''
        return self.encode(self.preprocess(sentences))


if __name__ == "__main__":
    sentences = ["abc", "def", "efg"]
    processor = CaptionEncoder()
    embeddings = processor.preprocess_encode(sentences)
    print(embeddings)