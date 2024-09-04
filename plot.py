import matplotlib.pyplot as plt
import numpy as np
import csv

endpoint_results = {}
with open('run_20240902_181026/experiment_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        endpoint_results[int(row['Parallel Requests'])] = {
            'combined_tps': float(row['Combined TPS']),
            'duration': float(row['Total Duration']),
            'price': float(row['Price per 1M Tokens'])
        }

gemini_results = {}
with open('gemini_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gemini_results[int(row['num_requests'])] = float(row['time'])

# Prepare data for plotting
parallel_requests = sorted(endpoint_results.keys())
endpoint_tps = [endpoint_results[r]['combined_tps'] for r in parallel_requests]
endpoint_duration = [endpoint_results[r]['duration'] for r in parallel_requests]
endpoint_price = [endpoint_results[r]['price'] for r in parallel_requests]

gemini_duration = [gemini_results.get(r, np.nan) for r in parallel_requests]

# Create the plot
fig, host = plt.subplots(figsize=(12, 8))

par1 = host.twinx()
par2 = host.twinx()

# Offset the right spine of par2 for better visibility
par2.spines['right'].set_position(('axes', 1.2))

host.set_xlabel("Number of Parallel Requests")
host.set_ylabel("Combined TPS")
par1.set_ylabel("Total Duration (seconds)")
par2.set_ylabel("Price per 1M tokens ($)")

p1, = host.plot(parallel_requests, endpoint_tps, "b-", label="Endpoint Combined TPS")
p2, = par1.plot(parallel_requests, endpoint_duration, "r-", label="Endpoint Duration")
p3, = par1.plot(parallel_requests, gemini_duration, "r--", label="Gemini Duration")
p4, = par2.plot(parallel_requests, endpoint_price, "g-", label="Endpoint Price per 1M tokens")
p5, = par2.plot(parallel_requests, [0.3] * len(parallel_requests), "g--", label="Gemini Price per 1M tokens")

host.set_xscale('log', base=2)
host.set_xticks(parallel_requests)
host.set_xticklabels([str(r) for r in parallel_requests])
plt.xticks(rotation=45)

# Adjust the Combined TPS axis (host)
max_tps = max(endpoint_tps)
host.set_ylim(0, max_tps + 200)  # Add a little padding
host.set_yticks(np.arange(0, max_tps + 200, 200))  # Set ticks every 200 TPS

# Adjust the duration axis (par1)
max_duration = max(max(endpoint_duration), max(gemini_duration))
par1.set_ylim(0, max_duration + 5)  # Add a little padding
par1.set_yticks(np.arange(0, max_duration + 5, 5))  # Set ticks every 5 seconds

# Adjust the price axis (par2)
par2.set_ylim(0, 2)  # Set y-axis limits from 0 to 2
par2.set_yticks(np.arange(0, 2.1, 0.1))  # Set ticks every 0.1 from 0 to 2
par2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))  # Format to 1 decimal place

lines = [p1, p2, p3, p4, p5]
host.legend(lines, [l.get_label() for l in lines], loc='upper left')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p4.get_color())

plt.title('Comparison Vertex AI endpoint and Gemini: TPS, Duration, and Price')
plt.tight_layout()
plt.savefig('endpoint_gemini_comparison.png', dpi=300, bbox_inches='tight')
