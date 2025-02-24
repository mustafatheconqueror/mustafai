import importlib
import tiktoken
"""
Learn re (tiktoken lib)

-
"""

def gpt_4_encoding_example():
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions

    vocab_size = tokenizer.n_vocab
    print("gpt-4 ", vocab_size)


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions
    text = (
        "Mustafa is a software engineer. <|endoftext|>"
    )

    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(token_ids)
    print(tokenizer.decode(token_ids))

    gpt_4_encoding_example()