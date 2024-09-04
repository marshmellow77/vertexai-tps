# vertexai-tps

A project to measure tokens per second (TPS) of models deployed on Vertex AI

The idea is to measure the TPS (and therefore calculate the serving costs) of a model deployed on an Vertex AI endpoint.

This is what the individual scripts do:

- [gemini_tps.py](gemini_tps.py): Sends `--num-requests` parallel requests to Gemini and measures response time
- [gemini_tps_loop.py](gemini_tps_loop.py): Sends an exponentially increasing number of parallel requests to Gemini and saves the results in a csv file
- [vertexai_endpoint_tps.py](vertexai_endpoint_tps.py): Sends `--num-requests` parallel requests to a Vertex AI Endpoint and measures response time and TPS
- [vertexai_endpoint_tps_loop.py](vertexai_endpoint_tps_loop.py): Sends an exponentially increasing number of parallel requests to a Vertex AI Endpoint and saves the results in a csv file
- [plot.py](plot.py): Creates a plot displaying Gemini results and Vertex AI Endpoint results

It is a good idea to run this in a virtual environment and pip install the required packages via `pip install -r requirements.txt`
