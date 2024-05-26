#!/bin/bash

INDEX_NAME="idx:table"

while true; do
  INFO_OUTPUT=$(redis-cli FT.INFO $INDEX_NAME)
  if [[ $INFO_OUTPUT == *"state: ready"* ]]; then
    echo "Index $INDEX_NAME is ready!"
    break
  else
    echo "Index is still building..."
  fi
  sleep 30
done