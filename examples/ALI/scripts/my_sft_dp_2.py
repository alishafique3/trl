# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "Pillow",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Train meta-llama/Llama-3.2-1B-Instruct on the trl-lib/Capybara dataset.

accelerate launch --config_file examples/ALI/accelerate_configs/single_gpu.yaml examples/ALI/scripts/my_sft_dp_2.py
"""


import os
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = model.num_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_as_messages(dataset, split="train"):
    
    def to_messages(example):
        user_content = example["instruction"]
        if example.get("input", "").strip():
            user_content += f"\n\n{example['input']}"
        
        return {"messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]}
        ]}
    
    return dataset.map(to_messages, remove_columns=dataset.column_names)


def main():
    # Load dataset
    train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    train_dataset = load_as_messages(train_dataset)
    train_dataset = train_dataset.select(range(1000))
    print(f"length of dataset: {len(train_dataset)}")
    print(train_dataset[0])


    # Load model
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained("examples/ALI/scripts/llama_custom_tokenizer") # Debug
    print_trainable_parameters(model)

    # # Debug: Apply chat template
    # def apply_chat_template(example):
    #     return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    # train_dataset = train_dataset.map(apply_chat_template, remove_columns=train_dataset.column_names)
    # print(train_dataset[0]['text'])

    # Train model
    training_args = SFTConfig(
        output_dir=f"examples/ALI/scripts/Llama-3.2-1B-Instruct",
        chat_template_path="examples/ALI/scripts/llama_custom_tokenizer",
        assistant_only_loss=True,
        eos_token="<|eot_id|>", # FOR Llama-3.2-1B-Instruct
        # bf16=True,
        # use_liger_kernel=True,
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        # max_length=8192,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        # dataset_num_proc=32,
        num_train_epochs=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()



