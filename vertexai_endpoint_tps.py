import argparse
from utils import make_prediction, calculate_metrics, warm_up_call

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using AI Platform.")
    parser.add_argument("--parallel_requests", type=int, default=8, help="Number of parallel requests")
    args = parser.parse_args()

    warm_up_call()

    total_tokens, time_taken, tokens_per_second = make_prediction(args.parallel_requests)
    avg_tps_per_request, combined_tps, avg_time_per_request, price_per_1m_tokens = calculate_metrics(total_tokens, time_taken, args.parallel_requests)

    print(f"Number of tokens: {total_tokens}")
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Average TPS per request: {avg_tps_per_request:.2f}")
    print(f"Combined TPS: {combined_tps:.2f}")
    print(f"Average time per request: {avg_time_per_request:.2f} seconds")
    print(f"Price per 1M tokens: ${price_per_1m_tokens:.4f}")