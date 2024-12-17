# this script is to test module import from the same parent folder
from utils import generate_uuid4
from text_processing.text_processor import remove_non_alphanumeric

print(generate_uuid4())
print(remove_non_alphanumeric('Hello World!'))