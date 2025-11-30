# Change Config files and Change variables
# run ipmitools then bash run4.sh then kill ipmitools

PROJECT_NAME="phd_deepspeed_zero3_8_bs_4_8b"
n_GPUs=8
CUDA_DEVICES="0,1,2,3,4,5,6,7"           
CONFIG_FILE="examples/ALI/accelerate_configs/deepspeed_zero3.yaml"
TRAIN_SCRIPT="examples/ALI/scripts/train_ft_sft_main10.py"

export PROJECT_NAME
export n_GPUs
export CUDA_DEVICES
export CONFIG_FILE
export TRAIN_SCRIPT

nvidia-smi --query-gpu=timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,temperature.gpu,temperature.memory,power.draw,power.limit,power.max_limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv -l 5 --id=$CUDA_DEVICES > ${PROJECT_NAME}_nvidia_smi.csv &

echo $! > ${PROJECT_NAME}_nvidia_monitor.pid

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=info TRANSFORMERS_NO_ADVISORY_WARNINGS=1 CUDA_VISIBLE_DEVICES=$CUDA_DEVICES accelerate launch --config_file $CONFIG_FILE $TRAIN_SCRIPT 2>&1 | tee ${PROJECT_NAME}_terminal.log

sleep 20

kill $(cat ${PROJECT_NAME}_nvidia_monitor.pid) 2>/dev/null
rm ${PROJECT_NAME}_nvidia_monitor.pid