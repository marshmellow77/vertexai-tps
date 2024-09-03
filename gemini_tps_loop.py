import asyncio
import argparse
import csv
from vertexai.generative_models import GenerativeModel, GenerationConfig
from utils import run_parallel_requests

async def main():
    parser = argparse.ArgumentParser(description="Run parallel requests to Gemini model")
    parser.add_argument("--max_exponent", type=int, default=3, help="Maximum exponent for number of parallel requests (2^max_exponent)")
    args = parser.parse_args()

    generation_config = GenerationConfig(
        max_output_tokens=1000,
    )
    model = GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

    prompt = "Tell me a bedtime story with at least 2000 words"

    results = []
    for exponent in range(args.max_exponent + 1):
        num_requests = 2 ** exponent
        print(f"Running experiment with {num_requests} parallel requests...")
        total_tokens, total_time, combined_tps = await run_parallel_requests(model, prompt, num_requests, generation_config)  # Updated
        results.append((num_requests, total_tokens, total_time))  # Updated
        print(f"Completed: {num_requests} requests in {total_time:.2f} seconds, Tokens: {total_tokens}")

    # Save results to CSV
    with open('gemini_flash_results_temp.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['num_requests', 'tokens', 'time'])  # Updated
        writer.writerows(results)

    print("Results saved to gemini_flash_results.csv")

if __name__ == "__main__":
    asyncio.run(main())