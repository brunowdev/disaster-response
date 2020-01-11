import re
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer

import spacy
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS

# Download some necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the us-eng model 
en_us_model = en_core_web_sm.load() # spacy.load('en_core_web_sm')

# Words related to questions on text
portuguese_question_tokens = [ 'qual', 'onde', 'quando', 'quem', 'como', 'porque', 'por que' ] 
question_tokens = portuguese_question_tokens + [ 'what', 'where', 'when', 'who', 'whom', 'how', 'why' ]

url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
url_placeholder = '_URL_'

nltk_stop_words = stopwords.words('english')
spacy_stop_words = list(STOP_WORDS)

all_stop_words = set(nltk_stop_words + spacy_stop_words)

lemmatizer = WordNetLemmatizer()

punctuation_tokenizer = RegexpTokenizer(r'\w+')

def create_placeholder_for_urls(text):
    return re.sub(url_regex, url_placeholder, text)

def words_without_stopwords(words):
    return list(filter(lambda w: w not in all_stop_words, words))

def remove_all_punctuation(text):
    return punctuation_tokenizer.tokenize(text)

def tokenize(text):
    lower_text = text.lower().strip()
    placeholder_text = create_placeholder_for_urls(lower_text)
    tokens = remove_all_punctuation(placeholder_text)
    only_relevant_words = words_without_stopwords(tokens)
    return list(map(lemmatizer.lemmatize, only_relevant_words))

def is_question(text):
    lower_text = en_us_model(text.lower())
    return lower_text[0].text in question_tokens or is_yes_or_no_question(lower_text)

def is_yes_or_no_question(text_model):
    return text_model[0].dep_ == 'aux'

def check_any_sentence_is_question(text):
    return any(filter(is_question, sent_tokenize(text.lower())))

class QuestionExtractor(BaseEstimator, TransformerMixin):

    def question_extractor(self, text):
        
        return check_any_sentence_is_question(text)

    def fit(self, x, y = None):
        return self

    def transform(self, X):
        return pd.DataFrame(pd.Series(X).apply(self.question_extractor))
    
class NumericDigitExtractor(BaseEstimator, TransformerMixin):

    def numeric_digit_extractor(self, text):
 
        sentence_list = sent_tokenize(text)
        
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            return False if len(pos_tags) == 0 else next(filter(lambda c: c[1] == 'CD', pos_tags), None) is not None

    def fit(self, x, y = None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.numeric_digit_extractor)
        return pd.DataFrame(X_tagged)