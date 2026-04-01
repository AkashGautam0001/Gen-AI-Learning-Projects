# import tiktoken
# encoder = tiktoken.get_encoding("cl100k_base")

# text = "Hello, how are you doing today?"
# tokens = encoder.encode(text)

# print(tokens)

# decoded_text = encoder.decode(tokens)
# print(decoded_text)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Hello, how are you doing today?"
tokens = tokenizer.encode(text)
print(tokens)
decoded_text = tokenizer.decode(tokens)
print(decoded_text)