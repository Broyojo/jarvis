from transformers import pipeline, set_seed
import time

generator = pipeline('text-generation', model='distilgpt2')
set_seed(42)
start = time.time()
result = generator("Hello, I’m a language model", max_length=128, num_return_sequences=1)
print(f"took {time.time() - start} seconds")
print(result)
