
# Total of categories allowed on this version
supported_categories = 36

def get_categories_name_from_text(text):
    return list(map(lambda cat: cat.split('-')[0], text.split(';')))

def get_categories_value_from_text(text):
    return list(map(lambda cat: int(cat.split('-')[1]), text.split(';')))