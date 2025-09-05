from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

msgs = [
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Say hi."}
]

out = tok.apply_chat_template(
    msgs,
    add_generation_prompt=True,              # <-- important
    return_dict=True,
    return_assistant_tokens_mask=True,
)
print("Has assistant_masks:", "assistant_masks" in out)
print("Assistant token count:", int(sum(out["assistant_masks"])))  # should be > 0
