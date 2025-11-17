# this adds the key from the .env file environment variable as a key.txt to the llm-confidentiality directory
#!/bin/bash

export $(grep -v '^#' .env | xargs)
echo $OPENAI_API_KEY > key.txt