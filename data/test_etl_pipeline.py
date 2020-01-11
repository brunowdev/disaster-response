import process_data as etl

import pandas as pd

# Sample string from the original dataset
valid_categories_sample_text = 'related-1;request-0;offer-0;aid_related-0;medical_help-0;medical_products-0;search_and_rescue-0;security-0;military-0;child_alone-0;water-0;food-0;shelter-0;clothing-0;money-0;missing_people-0;refugees-0;death-0;other_aid-0;infrastructure_related-0;transport-0;buildings-0;electricity-0;tools-0;hospitals-0;shops-0;aid_centers-0;other_infrastructure-0;weather_related-0;floods-0;storm-0;fire-0;earthquake-0;cold-0;other_weather-0;direct_report-0'

# Parsed categories - expected names
parsed_categories_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']

# Parsed categoires - expected values
parsed_categories_values = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def test_parse_column_names_length():
    assert len(etl.get_categories_name_from_text(valid_categories_sample_text)) == etl.supported_categories
        
def test_parse_column_names():
    assert etl.get_categories_name_from_text(valid_categories_sample_text) == parsed_categories_names

def test_parse_column_values():
    assert etl.get_categories_value_from_text(valid_categories_sample_text) == parsed_categories_values
    
def test_parse_column_values_length():
    assert len(etl.get_categories_value_from_text(valid_categories_sample_text)) == etl.supported_categories
    
def test_parse_dataframe_columns_with_values():
    
    df = pd.DataFrame({ 
            'id' : [10],
            'categories' : ['related-1;request-0;offer-0']
        })

    expected_categories = [ 'related', 'request', 'offer']
    expected_categories_values = [1, 0, 0]
    expected_columns = ['id', 'categories'] + expected_categories
    
    updated_df = etl.expand_categories_to_columns_with_values(df, 'categories', expected_categories)
    
    assert list(updated_df.columns) == expected_columns
    assert updated_df.related.max() == 1
    assert updated_df.request.max() == 0
    assert updated_df.offer.max() == 0