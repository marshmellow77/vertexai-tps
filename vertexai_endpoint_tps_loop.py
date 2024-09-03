import argparse
import os
import json
import csv
from datetime import datetime
from dotenv import load_dotenv
from utils import make_prediction, calculate_metrics, warm_up_call

# Load environment variables from .env file
load_dotenv()

def run_experiment(num_parallel_requests):
    try:
        num_tokens, time_taken, tokens_per_second = make_prediction(num_parallel_requests)
        
        avg_tps_per_request, combined_tps, avg_time_per_request, price_per_1m_tokens = calculate_metrics(num_tokens, time_taken, num_parallel_requests)
        
        return avg_tps_per_request, combined_tps, time_taken, avg_time_per_request, num_tokens, price_per_1m_tokens
    except Exception as e:
        print(f"Error during experiment with {num_parallel_requests} parallel requests: {str(e)}")
        return None

def save_checkpoint(results, completed_runs):
    checkpoint_path = os.path.join(RUN_FOLDER, 'checkpoint.json')
    with open(checkpoint_path, 'w') as f:
        json.dump({'results': results, 'completed_runs': completed_runs}, f)

def load_checkpoint():
    checkpoint_path = os.path.join(RUN_FOLDER, 'checkpoint.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        return data['results'], data['completed_runs']
    return {}, []

def run_experiments(max_exponent, num_runs):
    global RUN_FOLDER
    
    if not RUN_FOLDER:
        RUN_FOLDER = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(RUN_FOLDER, exist_ok=True)
        print(f"Created new run folder: {RUN_FOLDER}")
    else:
        print(f"Using specified run folder: {RUN_FOLDER}")
    
    warm_up_call()  # Perform warm-up call before starting experiments
    
    parallel_requests_range = [2**i for i in range(0, max_exponent + 1)]
    
    results, completed_runs = load_checkpoint()
    
    total_experiments = len(parallel_requests_range) * num_runs
    print(f"Starting experiments with {len(parallel_requests_range)} different parallel request counts, {num_runs} runs each...")

    for run in range(num_runs):
        for i, num_parallel_requests in enumerate(parallel_requests_range, 1):
            experiment_key = f"{run}_{num_parallel_requests}"
            if experiment_key in completed_runs:
                print(f"Skipping completed experiment: Run {run+1}, {num_parallel_requests} parallel requests")
                continue
            
            print(f"\nRun {run+1}, Experiment {i}/{len(parallel_requests_range)}: Running with {num_parallel_requests} parallel requests...")
            result = run_experiment(num_parallel_requests)
            
            if result is None:
                print(f"\nExperiment failed at Run {run+1}, Experiment {i}/{len(parallel_requests_range)} with {num_parallel_requests} parallel requests.")
                print(f"A checkpoint has been saved in the run folder: {RUN_FOLDER}")
                print(f"To resume the experiment, run the script with the following arguments:")
                print(f"python vertexai_endpoint_tps_loop.py --max_exponent {max_exponent} --num_runs {num_runs} --run_folder {RUN_FOLDER}")
                return
            
            avg_tps_per_request, combined_tps, total_duration, avg_time_per_request, total_tokens, price_per_1m_tokens = result
            
            if num_parallel_requests not in results:
                results[num_parallel_requests] = []
            
            results[num_parallel_requests].append({
                'avg_tps_per_request': avg_tps_per_request,
                'combined_tps': combined_tps,
                'total_duration': total_duration,
                'avg_time_per_request': avg_time_per_request,
                'total_tokens': total_tokens,
                'price_per_1m_tokens': price_per_1m_tokens
            })
            
            completed_runs.append(experiment_key)
            save_checkpoint(results, completed_runs)
            
            print(f"  Results for {num_parallel_requests} parallel requests:")
            print(f"  - Average individual TPS: {avg_tps_per_request:.2f}")
            print(f"  - Combined TPS: {combined_tps:.2f}")
            print(f"  - Total duration: {total_duration:.2f} seconds")
            print(f"  - Average time per request: {avg_time_per_request:.2f} seconds")
            print(f"  - Total tokens: {total_tokens}")
            print(f"  - Price per 1M tokens: ${price_per_1m_tokens:.4f}")
            print(f"Run {run+1}, Experiment {i}/{len(parallel_requests_range)} complete.")

    print("\nAll experiments completed.")
    
    generate_csv(results)

def generate_csv(results):
    csv_path = os.path.join(RUN_FOLDER, 'experiment_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Parallel Requests', 'Avg TPS per Request', 'Combined TPS', 'Total Duration', 'Avg Time per Request', 'Total Tokens', 'Price per 1M Tokens'])
        
        for num_requests, runs in results.items():
            avg_tps_per_request = sum(run['avg_tps_per_request'] for run in runs) / len(runs)
            combined_tps = sum(run['combined_tps'] for run in runs) / len(runs)
            total_duration = sum(run['total_duration'] for run in runs) / len(runs)
            avg_time_per_request = sum(run['avg_time_per_request'] for run in runs) / len(runs)
            total_tokens = sum(run['total_tokens'] for run in runs) / len(runs)
            price_per_1m_tokens = sum(run['price_per_1m_tokens'] for run in runs) / len(runs)
            
            writer.writerow([
                num_requests,
                f"{avg_tps_per_request:.2f}",
                f"{combined_tps:.2f}",
                f"{total_duration:.2f}",
                f"{avg_time_per_request:.2f}",
                total_tokens,
                f"{price_per_1m_tokens:.4f}"
            ])
    
    print(f"Raw data saved as '{csv_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--max_exponent', type=int, default=2, help='Maximum value of exponent for parallel_requests_range (max 8 for 256 parallel requests)')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs for each experiment')
    parser.add_argument('--run_folder', type=str, help='Specify a run folder to use or resume from')
    args = parser.parse_args()

    max_parallel_requests = 2**args.max_exponent
    if max_parallel_requests > 256:
        print(f"Error: The maximum number of parallel requests ({max_parallel_requests}) exceeds the API limit of 256.")
        print(f"Please choose a max_exponent value of 8 or less.")
        exit(1)

    RUN_FOLDER = args.run_folder

    print("Running experiments...")
    run_experiments(args.max_exponent, args.num_runs)
