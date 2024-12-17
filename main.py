from src.utils import generate_uuid4, generate_random_int
from src.text_processing.text_processor import remove_non_alphanumeric


# you may need to clear state to reflect changes in your modules
# Run/Clear/Clear state and outputs
print(generate_uuid4())
print(generate_random_int())
print(remove_non_alphanumeric("Hello World!"))