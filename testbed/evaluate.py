import argparse
import pandas as pd
from da_agent.evaluators.evaluation import Evaluator
import json
import os
import logging

def run_evaluation(output_dir, gold_dir, eval_json, result_dir, timeout_seconds):
    logging.basicConfig(level=logging.INFO)
    # Initialize the Evaluator with the provided directories
    evaluator = Evaluator(output_dir=output_dir, gold_dir=gold_dir, timeout_seconds=timeout_seconds)

    # Perform the evaluation
    results_infos = evaluator.evaluate(env_config=eval_json)
    num_results = len(results_infos)
    scores = [result['total_score'] for result in results_infos]
    finished = [result['finished'] for result in results_infos]
    domains = [result['domain'] for result in results_infos]
    df = pd.DataFrame({"score": scores, "finished": finished, "domain": domains})
    
    average_score = sum(scores) / num_results
    average_finished = sum(finished) / num_results
    results_json = {"num_results": num_results, "average_score": average_score, "results": results_infos, "average_finished": average_finished}
    print(f"Number of results: {num_results}")
    print(f"Average score: {average_score}")
    print(f"Average finished: {average_finished}")

    print("====================================")
    print(df.groupby("domain").agg({"score": "mean", "finished": "mean"}))
    
    # Ensure the directory exists before writing to the file
    os.makedirs(result_dir, exist_ok=True)
    file_name = output_dir.split("/")[-1]
    file_name = output_dir.split("/")[-2] if file_name == "" else file_name
    output_file = os.path.join(result_dir, os.path.basename(output_dir) + ".json")
    with open(output_file, 'w') as json_file:
        json.dump(results_json, json_file, indent=4)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run evaluations for NLP models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--gold_dir", type=str, default="gold", help="Directory containing gold standard files")
    parser.add_argument("--eval_json", type=str, default="tasks/dev.jsonl", help="JSON file with evaluation configurations")
    parser.add_argument("--result_dir", type=str, default="results", help="Directory to write evaluation results to")
    parser.add_argument("--timeout_seconds", type=int, default=120, help="Timeout for each evaluation in seconds")
    return parser.parse_args()

def main():
    args = parse_arguments()
    run_evaluation(args.output_dir, args.gold_dir, args.eval_json, args.result_dir, args.timeout_seconds)

if __name__ == "__main__":
    main()
