import os

# Define the directory path
directory_path = '/home/xshe/GE/'

# List all files in the directory
files = os.listdir(directory_path)

# Filter files that start with 'O'
files_to_delete = [file for file in files if file.endswith('png')]

# Delete the filtered files
for file in files_to_delete:
    os.remove(os.path.join(directory_path, file))

# Return the list of deleted files
files_to_delete
