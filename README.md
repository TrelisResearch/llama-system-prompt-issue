# Llama 1B System Prompt Issue Replicator

This repository contains a minimal replicator for an issue where the Llama 3.2 1B Instruct model (onnx-community/Llama-3.2-1B-Instruct-onnx-web-gqa) does not properly follow the system prompt, specifically when instructed to use search commands.

## Issue Description

The model is given a system prompt that instructs it to use search commands (e.g., `<search>query</search>`) when it needs information. However, when asked questions that would require searching (like weather queries), the model does not follow this instruction and instead provides direct responses without using the search command.

## Running the Test

Simply open `index.html` in Chrome/Brave browser (version 113+ with WebGPU support). The test will:

1. Load the ONNX web version of the model and tokenizer
2. Show the chat template being used
3. Display the formatted input (for debugging)
4. Generate and stream the model's response

## Expected vs Actual Behavior

Expected:
```
Let me check that for you.
<search>Paris France current weather</search>
```

Actual:
The model typically responds with direct statements or evasive responses without using the search command as instructed.

## Requirements

- Chrome/Chromium 113+ or equivalent with WebGPU support
- Modern browser with JavaScript enabled
- Sufficient memory to load the model (~1GB)

## Implementation Details

The replicator uses:
- transformers.js for model loading and inference
- WebGPU for hardware acceleration
- ONNX web runtime for model execution
- Streaming output for real-time response generation

## Debugging Information

The replicator shows:
- The exact chat template being used
- The formatted input sent to the model
- The complete model response
- Any errors that occur during execution

## Environment Information

- Model: onnx-community/Llama-3.2-1B-Instruct-onnx-web-gqa
- transformers.js version: latest from CDN
- Runtime: ONNX Web with WebGPU backend 