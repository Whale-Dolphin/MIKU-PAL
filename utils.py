import os

def sanitize_filename(filename):
    """
    Sanitizes a filename to prevent reading errors by removing or replacing invalid characters.
    """
    invalid_chars = '<>:"/\\|?*'
    sanitized_filename = filename

    for char in invalid_chars:
        sanitized_filename = sanitized_filename.replace(char, '_')

    if not sanitized_filename:
        raise ValueError("Filename cannot be empty after sanitization.")

    return sanitized_filename

def sanitize_filenames_in_directory(directory):
    """
    Sanitizes all filenames in a given directory.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            sanitized_name = sanitize_filename(file)
            original_path = os.path.join(root, file)
            sanitized_path = os.path.join(root, sanitized_name)
            os.rename(original_path, sanitized_path)