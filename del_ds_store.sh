#!/bin/bash

# Define the directory to search in. Replace "/path/to/directory" with the actual path.
SEARCH_DIR="."

# Use the find command to locate all .DS_Store files in the directory and its subdirectories,
# then delete each file found.
find "$SEARCH_DIR" -name '.DS_Store' -type f -exec rm -f {} +

echo "All .DS_Store files have been deleted."