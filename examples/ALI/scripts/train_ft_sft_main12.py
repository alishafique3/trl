from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
import os
import torch
import wandb
import argparse
import logging
from nvitop import Device, ResourceMetricCollector, collect_in_background
import time
import os
# from codecarbon import OfflineEmissionsTracker
import warnings
from accelerate import Accelerator
from accelerate import PartialState
from datetime import datetime
import sys



# Argument parser for command-line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")
# parser.add_argument("--model", type=str, required=True, help="Path to the pretrained model.")
# args = parser.parse_args()
# filename = os.path.basename(args.model)
filename = os.environ.get("PROJECT_NAME", "default")

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]= filename
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="false"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"



# Enable verbose loading
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["HF_HUB_VERBOSITY"] = "info"


# Master
accelerator = Accelerator()
# deepspeed.init_distributed()

if accelerator.is_main_process:
    print("This prints only on the main process!")
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("BF16 supported:", torch.cuda.is_bf16_supported())
    print("CUDA_DEVICES:", os.environ.get("CUDA_DEVICES", "None"))
    print("CONFIG_FILE:", os.environ.get("CONFIG_FILE", "None"))
    print("TRAIN_SCRIPT:", os.environ.get("TRAIN_SCRIPT", "None"))
    print("PROJECT_NAME:", os.environ.get("PROJECT_NAME", "None"))
    print("n_GPUs:", os.environ.get("n_GPUs", "None"))

# Suppress warnings in non-main processes
if not accelerator.is_main_process:
    warnings.filterwarnings("ignore")




############################################# Model and Tokenizer
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

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=128, #default 16
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)

# tokenizer = AutoTokenizer.from_pretrained("examples/ALI/scripts/llama_custom_tokenizer") # Debug

# if accelerator.is_main_process:
#     print_trainable_parameters(model) ##

# model = prepare_model_for_kbit_training(model)

# if accelerator.is_main_process:
#     print("\nPEFT\n")
# model = get_peft_model(model, peft_config)

# if accelerator.is_main_process:
#     print_trainable_parameters(model) ##

# tokenizer = AutoTokenizer.from_pretrained(args.model)
# eos_token = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.padding_side = "right"  # Fix the warning


################################################## Dataset
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

# Load dataset
train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
train_dataset = load_as_messages(train_dataset)
train_dataset = train_dataset.select(range(3200))

if accelerator.is_main_process:
    print(f"length of dataset: {len(train_dataset)}")
    print(train_dataset[0])

##################################################### Training Monitoring

logger = None


# Set up logging for nvitop GPU monitoring
if accelerator.is_main_process:
    file_path = filename + "_nvitop.log"
    
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
    else:
        print(f"{file_path} does not exist.")
    
    logging.basicConfig(filename=file_path, level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Function to log collected GPU metrics
    def on_collect(metrics):
        if logger is None:
            return False
        logger.info(metrics)  # Log GPU metrics every 5 seconds
        return True

    def on_stop(collector):
        if logger is not None:
            logger.info("GPU monitoring stopped.")

    collect_in_background(
        on_collect,
        ResourceMetricCollector(Device.cuda.all()),
        interval=5.0,
        on_stop=on_stop,
    )


start_time = time.time()
start_date = datetime.fromtimestamp(start_time)
if accelerator.is_main_process:
    print("\nTime Start\n")
    print(start_date)


##################################################### Training Setup

num_gpus = int(os.environ.get("n_GPUs", 1))

training_args = SFTConfig(
    output_dir=f"examples/ALI/scripts/Llama-3.1-8B-Instruct",
    run_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    chat_template_path= "meta-llama/Llama-3.1-8B-Instruct",  # "examples/ALI/scripts/llama_custom_tokenizer",
    # assistant_only_loss=True,
    eos_token="<|eot_id|>", # FOR Llama-3.1-8B-Instruct
    # bf16=True,
    # use_liger_kernel=True,
    gradient_checkpointing=False,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    # max_length=8192,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    # dataset_num_proc=32,
    num_train_epochs=10,
    save_strategy="no",
    #save_steps=10000,
    #save_total_limit=1,
    )

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    )

trainer.train()


# Stop NVITOP GPU monitoring (optional)
if accelerator.is_main_process:
    print("\nTime End\n")
    end_date = datetime.fromtimestamp(time.time())
    print(end_date)
    logger.info("Training completed. Stopping GPU monitoring.")


# Ensure all processes finish before exit
accelerator.wait_for_everyone()

if accelerator.is_main_process:
    print("✅ Training finished. Cleaning up...")
    # Manually finish wandb run
    wandb.finish() 
    # sys.exit(0)
    os._exit(0)  # Forcefully exit without waiting for background processes