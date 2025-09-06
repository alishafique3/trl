from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Custom chat template
# custom_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant: "

# Set custom template
# tokenizer.chat_template = custom_template

# Save to specific path
tokenizer.save_pretrained("examples/ALI/scripts/llama_custom_tokenizer")