import pytest
from src.train import get_data, train_model

# Test 1: Data Validation
def test_data_properties():
    """Tests if data is loaded correctly and has the right properties."""
    df = get_data()
    assert not df.empty
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    assert all(col in df.columns for col in expected_columns)
    assert df.isnull().sum().sum() == 0

# Test 2: Model Evaluation
def test_model_accuracy():
    """Tests if the model accuracy is above a 90% threshold."""
    # This test will also generate the confusion_matrix.png file
    accuracy = train_model()
    assert accuracy >= 0.90
