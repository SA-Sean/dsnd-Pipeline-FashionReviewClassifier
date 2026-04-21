# 1. Standard Libraries
import pandas as pd
import numpy as np

# 2. Scikit-Learn
from sklearn.base import BaseEstimator, TransformerMixin

# 3. NLP & AI Libraries
import spacy
from transformers import pipeline
from tqdm.auto import tqdm # a library that allows us to add a visual progress bar for long running operations



# Custom spaCy Lemmatization transformer
# using Joblib caching (ie: accepts a memory input object) to cahce the Lemmatisation function

class SpacyLemmatizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp, batch_size=256, memory=None):
        self.batch_size = batch_size
        self.nlp = nlp
        self.memory = memory  # pass in memory object for caching, explicitly caching the Lemmatisation function
        self.feature_names_in_ = None 

    def fit(self, X, y=None):
        # 1. Capture names from DataFrame or Series
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        elif hasattr(X, "name"):
            self.feature_names_in_ = np.array([X.name], dtype=object)
        else:
            self.feature_names_in_ = np.array(["text_input"], dtype=object)
        return self
    
    def transform(self, X):
        # handle input X and ensure we get a string only series
        if isinstance(X, pd.DataFrame):
            text_series = X.iloc[:,0].astype(str)
        else:
            text_series = pd.Series(np.array(X).ravel()).astype(str)

        texts = text_series.to_list()

        # execute cached logic using the cache wrapper as below
        if self.memory is not None:
            cached_logic = self.memory.cache(self._run_lemmatization, ignore=['self'])
            # pass model name/meta to ensure cache invalidates if nlp model changes
            lemmatized_texts = cached_logic(texts, self.nlp.meta['name'], self.batch_size)
        else:
            lemmatized_texts = self._run_lemmatization(texts, self.nlp.meta['name'], self.batch_size)

        return pd.Series(lemmatized_texts, index=text_series.index, name=self.feature_names_in_[0])

    def _run_lemmatization(self, texts, nlp_name, batch_size):

        """The heavy lifting logic (Spacy pipe) that gets cached."""

        print(f"\n[Cache Miss] Lemmatizing {len(texts)} texts using {nlp_name}...")
        
        lemmatized_texts = []

        # use nlp.pipe which should use optimised batch processing and be faser than nlp(text)
        for doc in tqdm(self.nlp.pipe(texts, batch_size=batch_size), total=len(texts), desc='Lemmatizing...'):
            # join lemmas with a space
            lemmatized_texts.append(' '.join([token.lemma_ for token in doc]))
            
        return lemmatized_texts

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.feature_names_in_
        return np.array([f"{names[0]}_lemmatized"], dtype=object)