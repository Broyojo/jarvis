"""
General Algorithm:

while listening:
    if wake signal is detected:
        while speaking:
            record and transcribe audio
        send text to chatgpt along with conversation
        collect streaming output and stream to TTS
        add streaming output to running conversation

Pipeline:
Wake Signal -> Audio from micrphone -> Whisper -> ChatGPT -> Output -> TTS
"""


import openai
import tiktoken


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


# messages = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant who can use external tools.",
#     },
#     {
#         "role": "user",
#         "content": "What is 234876 / 123?",
#     },
#     {
#         "role": "assistant",
#         "content": "234876 divided by 123 equals [calculator 234876 / 123]",
#     },
#     {
#         "role": "user",
#         "content": "What's the weather today?",
#     },
#     {
#         "role": "assistant",
#         "content": "The weather today is [weather [location]]",
#     },
# ]

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
]

while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})

    response = ""
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    ):
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)
            response += delta["content"]
    messages.append({"role": "assistant", "content": response})
    print()
    print(
        "Number of tokens so far:",
        num_tokens_from_messages(messages),
    )
