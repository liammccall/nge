#!/bin/bash

API_URL="http://localhost:8000/api/chat"
TEMPLATE_FILE="template.json"

if ! command -v jq &> /dev/null; then
    echo "jq is required (install: sudo apt install jq)"
    exit 1
fi

USER_MSG=$(jq -r '.user' "$TEMPLATE_FILE")

curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{\"user\": \"$USER_MSG\"}" | jq