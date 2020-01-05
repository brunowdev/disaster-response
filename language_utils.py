#!/usr/bin/python
# -*- coding: utf-8 -*-
import nltk

nltk.download('words')
nltk.download('wordnet')

from nltk.corpus import words
from nltk.stem import WordNetLemmatizer

import re

english_words = words.words()
lemmatizer = WordNetLemmatizer()

additional_words = ['website']

valid_words = english_words + additional_words

def lemmatize_word(word):
    return lemmatizer.lemmatize(word)

def remove_ponctuation(text):
    return re.sub(f'[!.:]', '', text)

def has_any_word(text):
    
    """
        Why split and not word_tokenize
        The nltk function causes many invalid texts beeing parsed as valid
        Example: 
            >>> word_tokenize('#$#@#!@&##!@me')
            ['#', '$', '#', '@', '#', '!', '@', '&', '#', '#', '!', '@', 'me']
            Will have a valid word, since me its a valid one
    """
    
    lower_text = text.lower()

    split_words = lower_text.split(' ')
    
    text_words = map(remove_ponctuation, split_words)
    
    lemm_words = map(lemmatize_word, text_words)
   
    return any(w in valid_words for w in lemm_words)