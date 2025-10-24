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
import deepspeed
from deepspeed.profiling.flops_profiler import FlopsProfiler
import argparse
import logging
from nvitop import Device, ResourceMetricCollector, collect_in_background
import time
import os
from codecarbon import OfflineEmissionsTracker
import warnings
from accelerate import Accelerator
from accelerate import PartialState
from dotenv import load_dotenv
from datetime import datetime
import sys



# Argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")
parser.add_argument("--model", type=str, required=True, help="Path to the pretrained model.")
args = parser.parse_args()
filename = os.path.basename(args.model)

load_dotenv()  # take environment variables from .env.

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]= filename + "_BNL"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="false"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


# Master
accelerator = Accelerator()
# deepspeed.init_distributed()

if accelerator.is_main_process:
    print("This prints only on the main process!")

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

# device_map='auto' # for DP and running with `python test_sft.py`
device_map="DDP" # for DDP and running with `accelerate launch test_sft.py`

if device_map == "DDP":
    device_string = PartialState().process_index
    device_map={'':device_string}

if accelerator.is_main_process:
    print("\nMODEL\n")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=bnb_config,
    use_cache=False,
    device_map = device_map,
)
if accelerator.is_main_process:
    print_trainable_parameters(model) ##

model = prepare_model_for_kbit_training(model)


if accelerator.is_main_process:
    print("\nPEFT\n")
model = get_peft_model(model, peft_config)

if accelerator.is_main_process:
    print_trainable_parameters(model) ##

tokenizer = AutoTokenizer.from_pretrained(args.model)
eos_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"  # Fix the warning


################################################ Compute Loss

class FLOPsSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prof = FlopsProfiler(self.model)  # Attach profiler to model
        self.step_n = 0  # Move step_n to instance variable
        self.print_profile = True  # Flag to print profiling output
        self.profiling_done = False  # Flag to disable profiling after first run

        # Dynamically calculate when to start profiling (50% of total training steps)
        total_train_steps = (len(self.train_dataset) // (self.args.per_device_train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps)) * self.args.num_train_epochs
        self.profile_step = total_train_steps // 2  # Set profiling step to 50% of total steps
        print(self.profile_step)
        print(len(self.train_dataset))
        print(self.args.per_device_train_batch_size)
        print(self.args.gradient_accumulation_steps)
        print(self.accelerator.num_processes)
        print("################################################")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # print("Hello")
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        
        # Start profiling at the defined step
        if self.step_n == self.profile_step and not self.profiling_done:
            self.prof.start_profile()
        
        outputs = model(**inputs)
        
        # End profiling and print output
        if self.step_n == self.profile_step and not self.profiling_done:
            self.prof.stop_profile()
            flops = self.prof.get_total_flops()
            macs = self.prof.get_total_macs()
            params = self.prof.get_total_params()
            if self.print_profile:
                self.prof.print_model_profile(profile_step=self.step_n, module_depth=-1, top_modules=3, detailed=True, output_file=filename+"_deepspeed.txt")
            self.prof.end_profile()

            self.profiling_done = True

        # Only increment `step_n` if profiling is not finished
        if not self.profiling_done:
            self.step_n += 1
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss



################################################## Dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        if len(input_text) >= 2:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Input:
            {input_text}
            
            ### Response:
            {response}
            '''
        else:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Response:
            {response}
            '''
        output_text.append(text + eos_token)

    return output_text


##################################################### Training Monitoring

# Set up CodeCarbon
if accelerator.is_main_process:
    tracker = OfflineEmissionsTracker(project_name="training", measure_power_secs=5, country_iso_code="USA", pue=1.3, region="california", allow_multiple_runs=False, output_file=filename+"_emission.csv")

# Set up logging for nvitop GPU monitoring
if accelerator.is_main_process:
    file_path = filename + ".log"
    
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

if accelerator.is_main_process:
    print("\nTRAINING\n")
    tracker.start_task("Training")

start_time = time.time()
start_date = datetime.fromtimestamp(start_time)
if accelerator.is_main_process:
    print("\nTime Start\n")
    print(start_date)


##################################################### Training Setup
training_args = SFTConfig(
    max_seq_length=1024,
    output_dir=filename,
    run_name= str(start_date),
    packing=False,
    logging_steps=1,
    warmup_steps = 5,
    # max_steps = 20,
    num_train_epochs = 1,
    learning_rate = 2e-4,
    per_device_train_batch_size = 4, #default 4
    gradient_accumulation_steps = 1, #default 2
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = {"use_reentrant": False}, #must be false for DDP
)



trainer = FLOPsSFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    # peft_config=peft_config,
    args=training_args,
)


##################################################### Training

trainer.train()

# Stop CodeCarbon Task
if accelerator.is_main_process:
    tracker.stop_task()


# Stop NVITOP GPU monitoring (optional)
if accelerator.is_main_process:
    print("\nTime End\n")
    end_date = datetime.fromtimestamp(time.time())
    print(end_date)
    logger.info("Training completed. Stopping GPU monitoring.")



# Stop CodeCarbon
if accelerator.is_main_process:
    emissions = tracker.stop()
    print("=" * 30)
    print(tracker.final_emissions_data)
    
    
    print(f"Emissions : {1000 * emissions} g CO₂")
    
    for task_name, task in tracker._tasks.items():
        print(
            f"Emissions :{1000 * task.emissions_data.emissions} g CO₂ for task {task_name}"
        )
        print(f"\tEnergy    CPU:{1000 * task.emissions_data.cpu_energy}Wh   GPU:{1000 * task.emissions_data.gpu_energy}Wh    RAM: {1000 * task.emissions_data.ram_energy}Wh"
             )
        print(
            f"\tPower CPU:{task.emissions_data.cpu_power:.0f}W   GPU:{task.emissions_data.gpu_power:.0f}W   RAM {task.emissions_data.ram_power:.0f}W" + f" during (data duration) {task.emissions_data.duration} seconds."
        )


# Ensure all processes finish before exit
accelerator.wait_for_everyone()

if accelerator.is_main_process:
    print("✅ Training finished. Cleaning up...")
    # 🔥 Manually finish wandb run
    wandb.finish() 
    # sys.exit(0)
    os._exit(0)  # Forcefully exit without waiting for background processes