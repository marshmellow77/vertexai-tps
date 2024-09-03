import asyncio
import argparse
from vertexai.generative_models import GenerativeModel, GenerationConfig
from utils import run_parallel_requests 

async def main():
    parser = argparse.ArgumentParser(description="Run parallel requests to Gemini model")
    parser.add_argument("--num_requests", type=int, default=8, help="Number of parallel requests to make")
    args = parser.parse_args()

    generation_config = GenerationConfig(
        max_output_tokens=1000,
    )
    model = GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

    prompt = "Tell me a bedtime story with at least 2000 words"
    num_requests = args.num_requests

    total_tokens, total_time, combined_tps = await run_parallel_requests(model, prompt, num_requests, generation_config)  # Updated

    print("\nSummary:")
    print(f"{'Requests':^10}|{'Tokens':^10}|{'Time (s)':^10}|{'TPS':^10}")
    print("-" * 44)
    print(f"{num_requests:^10}|{total_tokens:^10}|{total_time:^10.2f}|{combined_tps:^10.2f}")

if __name__ == "__main__":
    asyncio.run(main())