import etl_pipeline as etl

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