NU nano 6.2                                                                                       new.sh                                                                                                 
#!/bin/sh

# Navigate to the directory containing the library file
cd /usr/local/lib/python3.10/dist-packages/langchain_community/document_loaders

# Name of the library file
LIBRARY_FILE="llmsherpa.py"

# Line to insert into the library file
INSERT_LINE='            "page_number": chunk.page_idx,'

# Pattern to match where the line should be inserted
MATCH_PATTERN='"chunk_number": chunk_num,'

# Insert the line into the library file after the matching pattern
sed -i "/$MATCH_PATTERN/a\\
$INSERT_LINE" "$LIBRARY_FILE"

# Navigate back to the app/inference directory
cd /app/inference

# Start Python HTTP server on port 8001
python3 -m http.server 8001

