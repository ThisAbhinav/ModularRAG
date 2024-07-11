#!/bin/bash

# Function to run the main tasks
run_main_tasks() {
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

    # Update config.json to set "use_cache" to false
    jq '.use_cache = false' config.json > tmp.$$.json && mv tmp.$$.json config.json

    # Run the first instance of the Chainlit app
    chainlit run app.py -w

    # Update config.json to set "use_cache" to true
    jq '.use_cache = true' config.json > tmp.$$.json && mv tmp.$$.json config.json

    # Run the second instance of the Chainlit app
    chainlit run app.py -w
}

# Run the main tasks
run_main_tasks

# Detach and return to the prompt
disown

