from pycaret.datasets import get_data
from pycaret.classification import setup, compare_models

# Load a sample dataset
data = get_data('iris')

# Initialize the PyCaret setup
clf_setup = setup(data=data, target='species', verbose=False, session_id=123)

# Compare models and display the best one
best_model = compare_models()
print(f"Best Model: {best_model}")