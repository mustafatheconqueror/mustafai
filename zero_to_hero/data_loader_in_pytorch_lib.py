""""
Data loader module

"""


import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True,
                          num_workers=0):

    tokenizer = tiktoken.get_encoding("cl100k_base")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


if "__main__" == __name__:

    # my input text
    input_text = "Mustafa is a software engineer.<|endoftext|>"


    #create gpt-4 tokenizer
    myTokenizer = tiktoken.get_encoding("cl100k_base")

    #encode
    token_ids = myTokenizer.encode(input_text, allowed_special={"<|endoftext|>"})

    #decode
    for token_id in token_ids:
        print(token_id, myTokenizer.decode_single_token_bytes(token_id))

    # create data loader

    dataloader = create_dataloader_v1(input_text, batch_size=1, max_length=5, stride=1, shuffle=False)

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    #The first_batch variable contains two tensors:
    # the first tensor stores the input token IDs, and the second tensor
    # stores the target token IDs.
