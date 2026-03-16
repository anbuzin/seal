#!/bin/bash
# Test the chat endpoint

echo "Testing /api/health endpoint..."
curl -s http://localhost:3000/api/health
echo ""
echo "---"

echo "Testing /api/chat endpoint..."
curl -s -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one word"
      }
    ]
  }'
echo ""
