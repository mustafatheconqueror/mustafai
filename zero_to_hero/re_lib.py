import re

"""
Learn re (Reguler expression library)

- re.split() --> is simple split for text we need to specify pattersn
    - re.split(r'(\s)', text)
    - re.split(r'([,.]|\s)', text)
    - re.split(r'([,.:;?_!"()\']|--|\s)', text)
"""

def simple_split(text):
    splitted_text = re.split(r'(.)', text)
    return splitted_text

def good_split(text):
    splitted_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return splitted_text



if __name__ == "__main__":
    simple_text = "Mustafa is, software engineer."

    simple_splitted_text = simple_split(simple_text)
    print(simple_splitted_text)

    good_splitted_text = good_split(simple_text)
    print(good_splitted_text)