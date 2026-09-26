model_name=$1

# check gpu memory and wait until it is available
while true; do
    for gpu in 0 2; do
        gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu)
        if [ "$gpu_memory" -gt 20000 ]; then
            export CUDA_VISIBLE_DEVICES=$gpu,$((gpu+1))
            break 2
        fi
    done
    sleep 10
done

nohup python -u generate_test_cases.py --method_name=Exh --experiment_name=$model_name --behaviors_path=./data/behavior_datasets/harmbench_behaviors_text_non_copyright.csv --save_dir=./results/Exh1/$model_name/test_cases >>results/beast.$model_name.log 2>&1&