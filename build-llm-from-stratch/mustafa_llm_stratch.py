import re
import torch
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

class MustafaTokenizerV1:
    """This is a simple word tokenizer."""
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {}

        for s, i in vocab.items():
            self.int_to_str[i] = s


    def encode(self, text_to_be_tokenized):
        # step 1- split et ve texti tek tek word word tokenlara ayır.
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text_to_be_tokenized)
        new_preprocessed = []
        ids = []
        #step2- split yaptıktan sonra whitespace tokenları silmek istiyorum.
        for token in preprocessed:
            stripped_item = token.strip()
            if stripped_item:
                new_preprocessed.append(stripped_item)

        print("Original", new_preprocessed)

        for token in new_preprocessed:
            ids.append(self.str_to_int[token])

        return ids



    def decode(self, token_ids):
        text = []
        for token_id in token_ids:
            single_word = self.int_to_str[token_id]
            single_word = re.sub(r'\s+([,.?!"()\'])', r'\1', single_word)
            text.append(single_word)
        return text

def practice_everything():
    simple_text = "Mustafa is, software engineer."
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', simple_text)
    all_words = sorted(set(preprocessed))


    vocab = {}

    for index, value in enumerate(all_words):
        vocab[value] = index

    mustafa_tokenizer_v1 = MustafaTokenizerV1(vocab)
    token_ids =  mustafa_tokenizer_v1.encode(simple_text)

    print("Encoded: ", token_ids)
    print("Decoded: ", mustafa_tokenizer_v1.decode(token_ids))

    #create token embedding.
    input_ids = torch.tensor([2, 3, 5, 1]) #we have 4 input token

    vocab_size = 6 #we have only 6 word
    output_dimension = 3

    embedding_layer = torch.nn.Embedding(vocab_size, output_dimension)
    print(embedding_layer.weight)


def simplifed_Attention_mechanism():
    #word is: your journey starts with one step.

    #after load input embedding (here for just sampling)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )

    # Journey
    query_token = inputs[1]

    # initialize as empty tensor.
    attention_scores = torch.empty(inputs.shape[0])

    #her input vector ile query tokenı dotproduct iler çarpacağız ki
    #ki yakınlıklarını anlayabilmek için

    #x_i is input token
    for i, x_i in enumerate(inputs):
        attention_scores[i] = torch.dot(x_i ,query_token)

    #apply classicalnormalization.
    attn_weights_2_tmp = attention_scores / attention_scores.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())

    # appyl pytorch softwax normalization
    attention_weights = torch.softmax(attention_scores, dim=0)
    print("Attention weights softmax:", attention_weights)
    print("Sum softmax:", attention_weights.sum())

    #create context vector.
    context_vector_of_2 = torch.zeros(query_token.shape)
    for i, x_i in enumerate(inputs):
        context_vector_of_2 += attention_weights[i] * x_i

    print("context vector of 2 : ", context_vector_of_2)


if __name__ == "__main__":

    with open(file="data-sets/the-verdict.txt", mode="r", encoding="utf-8") as file:
        raw_text = file.read()
        raw_text = raw_text[:99]

    #practice_everything()
    simplifed_Attention_mechanism()
