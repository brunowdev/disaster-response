#!/usr/bin/python
# -*- coding: utf-8 -*-

import language_utils as language

def test_lemmatize_word():
    
    assert language.lemmatize_word('sections') == 'section'
    
def test_remove_ponctuation():
    
    text_to_parse = '@Only: some ponctuations should be removed!.'
    expected_result = '@Only some ponctuations should be removed'
    
    assert language.remove_ponctuation(text_to_parse) == expected_result

def test_has_any_word_with_valid_text():
    
    text_to_parse = 'Weather update - a cold front from Cuba that could pass over Haiti'

    assert language.has_any_word(text_to_parse) is True
    
def test_has_any_word_with_invalid_text():
    
    text_to_parse = '0H 8@@ @2x @e8x3 xr'

    assert language.has_any_word(text_to_parse) is False