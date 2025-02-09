import { env, AutoTokenizer, AutoModelForCausalLM, TextStreamer } from '@huggingface/transformers';

// Skip local model check since we're downloading from HF Hub
env.allowLocalModels = false;

async function testSystemPrompt() {
    const model_id = "onnx-community/Llama-3.2-1B-Instruct-onnx-web-gqa";
    
    try {
        // Initialize tokenizer and model
        console.log('Loading tokenizer...');
        const tokenizer = await AutoTokenizer.from_pretrained(model_id);
        
        console.log('Loading model...');
        const model = await AutoModelForCausalLM.from_pretrained(model_id, {
            dtype: "q4f16",
            device: "webgpu"
        });
        
        // System prompt that should be followed
        const system_prompt = `You are a helpful AI assistant that can search the web. When you need information to answer a question, use the search command like this:

<search>your search query</search>

For example:
User: What's the weather in Dublin?
Assistant: Let me check that for you.
<search>Dublin Ireland current weather</search>

Keep your responses concise and focused on the information needed.`;
        
        // Test messages
        const messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What's the weather in Paris?"}
        ];
        
        // Format with chat template
        console.log('\nApplying chat template...');
        console.log('Current chat template:', tokenizer.chat_template);
        
        // Get readable format first for debugging
        const readable_input = tokenizer.apply_chat_template(messages, {
            add_generation_prompt: true,
            tokenize: false
        });
        console.log('\nFormatted input (readable):');
        console.log(readable_input);
        
        // Generate response
        const inputs = tokenizer.apply_chat_template(messages, {
            add_generation_prompt: true,
            return_tensors: "pt"
        });
        
        console.log('\nGenerating response...');
        let currentResponse = '';
        const streamer = new TextStreamer(tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (token) => {
                currentResponse += token;
                console.log(token);
            }
        });
        
        await model.generate({
            ...inputs,
            max_new_tokens: 100,
            do_sample: true,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
            streamer,
        });
        
        console.log('\nFinal response:');
        console.log(currentResponse);
        
    } catch (error) {
        console.error('Error:', error);
    }
}

// Run the test
testSystemPrompt(); 