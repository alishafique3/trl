from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("examples/ALI/scripts/llama_custom_tokenizer")

msgs = [
  {"role":"user","content":"hi"},
  {"role":"assistant","content":"Hello"}
]

out = tokenizer.apply_chat_template(
    msgs,
    add_generation_prompt=False,
    return_dict=True,
    return_assistant_tokens_mask=True,
)

# Print templated prompt
print(tokenizer.decode(out["input_ids"]))
print(out)
print("\nHas assistant_masks:", "assistant_masks" in out)
print("Assistant token count:", int(sum(out["assistant_masks"])))