import os
import time
from google.cloud import aiplatform
from dotenv import load_dotenv
from transformers import AutoTokenizer
import asyncio


# Load environment variables from .env file
load_dotenv()

# Constants
PROJECT_ID = os.getenv("PROJECT_ID")
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
REGION = os.getenv("REGION")
MAX_TOKENS = 1000

# Ensure environment variables are set
if not PROJECT_ID or not ENDPOINT_ID or not REGION:
    raise ValueError("Please set the PROJECT_ID, ENDPOINT_ID, and REGION environment variables in the .env file.")

# Initialize gemma tokenizer
gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

def generate(prompt, num_parallel_requests):
    instances = [{
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    } for _ in range(num_parallel_requests)]
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    response = endpoint.predict(instances=instances)
    return response

def make_prediction(num_parallel_requests, prompt="Write me a very long story with at least 10000 words"):
    start_time = time.time()
    response = generate(prompt, num_parallel_requests)
    end_time = time.time()
    
    total_tokens = 0
    for prediction in response.predictions:
        tokens = gemma_tokenizer.encode(prediction)
        total_tokens += len(tokens) - 1
            
    time_taken = end_time - start_time
    tokens_per_second = total_tokens / time_taken if time_taken > 0 else float('inf')
    
    return total_tokens, time_taken, tokens_per_second

def calculate_metrics(num_tokens, time_taken, num_parallel_requests):
    avg_tps_per_request = (num_tokens / num_parallel_requests) / time_taken
    combined_tps = num_tokens / time_taken
    avg_time_per_request = time_taken / num_parallel_requests
    
    # Calculate price per 1M tokens
    hourly_price = 5.5  # $5.5 per hour
    price_per_1m_tokens = (hourly_price / 3600) / (combined_tps / 1_000_000)
    
    return avg_tps_per_request, combined_tps, avg_time_per_request, price_per_1m_tokens

def warm_up_call():
    print("Performing warm-up call to the LLM...")
    prompt = "This is a warm-up call."
    generate(prompt, 1)  # Single request for warm-up
    print("Warm-up call completed.")
    
async def run_single_request(model, prompt, generation_config):
    start_time = time.time()
    response = await model.generate_content_async(prompt)
    end_time = time.time()
    elapsed_time = end_time - start_time
    tokens = response.usage_metadata.candidates_token_count
    return tokens, elapsed_time

async def run_parallel_requests(model, prompt, num_requests, generation_config):
    start_time = time.time()
    tasks = [run_single_request(model, prompt, generation_config) for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    total_tokens = sum(tokens for tokens, _ in results)
    total_time = end_time - start_time
    combined_tps = total_tokens / total_time

    return total_tokens, total_time, combined_tps