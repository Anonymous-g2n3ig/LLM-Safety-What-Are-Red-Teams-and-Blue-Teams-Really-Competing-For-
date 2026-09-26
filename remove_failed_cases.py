# This is a script to conclude the results of a given experiment. 
import os, shutil

def del_results(behavioral_dataset, experimental_results):
    """
    This function takes in a behavioral dataset and experimental results,
    and deletes the failed cases from the experimental results.

    Parameters:
    behavioral_dataset (DataFrame): The dataset containing behavioral data.
    experimental_results (dict): A dictionary containing the results of the experiment.

    behavioral_dataset format:
    ```bash
    $ cat $behavioral_dataset
    Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID
    ...
    ```

    experimental_results format:
    ```bash
    $ ls $experimental_results/test_cases/test_cases_individual_behaviors
    BehaviorIDx
    BehaviorIDy
    BehaviorIDz
    ...

    $ cat $experimental_results/test_cases/test_cases_individual_behaviors/BehaviorIDx/logs.json
    {
        BehaviorID: [
            [
                "test_case": Behaviorx,
                "number_of_tried_prompts": ${number_of_tried_prompts:int},
                "successful_attack": ${successful_attack:bool},
                "running_time": ${running_time:str},
                "running_seconds": ${running_seconds:float},
                "response": ${response:str}
            ]
        ]
    }
    """
    
    # read the BehaviorID from the behavioral dataset
    import pandas as pd
    behavioral_df = pd.read_csv(behavioral_dataset)
    behavior_ids = behavioral_df['BehaviorID'].tolist()

    # iterate through the behavior_ids and get the corresponding results from experimental_results
    concluded_results = []
    for behavior_id in behavior_ids:
        # construct the path to the logs.json file for the current behavior_id
        logs_path = f"{experimental_results}/test_cases/test_cases_individual_behaviors/{behavior_id}"
        # read the logs.json file
        import json
        with open(os.path.join(logs_path, "logs.json"), 'r') as f:
            logs = json.load(f)
            if logs[behavior_id][0][0]["test_case"] is None:
                print(logs[behavior_id][0][0]["response"])
                # shutil.rmtree(logs_path)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Conclude the results of a given experiment.")
    parser.add_argument("--behavioral_dataset", type=str, help="Path to the behavioral dataset CSV file.", default="data/behavior_datasets/harmbench_behaviors_text_non_copyright.csv")
    parser.add_argument("--model", type=str, help="The model name used in the experiment.", default="mistral_7b_v2")
    parser.add_argument("--method", type=str, help="The method used to attack the model.", default="Exh1")
    args = parser.parse_args()


    # Construct the path to the experimental results based on the model and method    
    experimental_results = f"results/{args.method}/{args.model}"
    del_results(args.behavioral_dataset, experimental_results)