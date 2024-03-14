from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Function to generate response given user input and chat history
def generate_response(user_input, history=[]):
    # Tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors="pt"
    )

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)

    # Generate a response
    response_ids = model.generate(
        bot_input_ids, max_length=4000, pad_token_id=tokenizer.eos_token_id
    ).tolist()

    # Convert the response tokens to text
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return response_text, bot_input_ids.tolist()

# Define the Gradio interface
title = "GreenBot"
description = "A friendly chatbot - Free and easy to use"
examples = [
    ["How was your day?"],
    ["What's your favorite color?"],
    ["Tell me a joke!"],
    ["Can you recommend a good movie?"],
    ["What do you think about artificial intelligence?"],
]

def chatbot_interface(input_text, chat_history=[]):
    response_text, new_chat_history = generate_response(input_text, chat_history)
    return response_text, new_chat_history

# Launch the interface
gr.Interface(
    fn=chatbot_interface,
    inputs=["text", "state"],
    outputs=["text", "state"],
    title=title,
    description=description,
    examples=examples,
    theme="abidlabs/pakistan",
).launch()
