import openai
import time

with open("key.txt", "r") as f:
    openai.api_key = f.read().strip()

start = time.time()
response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Write a story:",
    max_tokens=20,
    temperature=0
)

print(f"took {time.time() - start} seconds")
print(response)
