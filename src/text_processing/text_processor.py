def remove_non_alphanumeric(input_string:str):
    return ''.join(c for c in input_string if c.isalnum())