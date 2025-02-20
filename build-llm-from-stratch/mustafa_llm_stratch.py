import re

class SimpleTokenizerV1:
    #Every Tokenizer has encode and decode method
    # we need to think how to handle not exist words or swhitespades or etc.
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str =  self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

def example_simple_usage_of_re_lib():
    text = "Hello, world. This, is a test."
    result1 = re.split(r'(\s)', text)
    result2 = re.split(r'([,.]|\s)', text)

    #remove whitespaces
    result3 = [item for item in result2 if item.strip()]
    print(result3)

def step_1_create_simple_custom_tokens_using_re_lib(raw_text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:1])
    step_2_Create_simple_token_id(preprocessed)
def step_2_Create_simple_token_id(preprocessed):
   # burada önce kelimeler için bir set oluşturdum. duplikasyonun önüne geçmek için daha sonra da bunu alfabetik olacak şekilde sıraladım.
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)
    #buradaki vocabulary bizim key-value pairleri temsil ediyor. ve aslında her bir token'a bir Id atamasını tamamlamış oluyoruz.
    # the dictionary contains individual tokens associated with unique inteteger labels.
    vocab = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 2:
            break
    step_3_use_simple_tokenizer(vocab)

def step_3_use_simple_tokenizer(vocab):

    tokenizer = SimpleTokenizerV1(vocab)

    text = """"It's the last he painted, you know," 
               Mrs. Gisburn said with pardonable pride."""

    ids = tokenizer.encode(text) #encode
    print(ids)

    print(tokenizer.decode(ids)) #you can decode also


def practice_everything():
    print("practice start.")


if __name__ == "__main__":

    with open(file="data-sets/the-verdict.txt", mode="r", encoding="utf-8") as file:
        raw_text = file.read()
        raw_text = raw_text[:99]


