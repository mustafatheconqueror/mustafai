"""
Burada default python library'lerinin neler yaptığını öğreneceğiz.
to be investigate libraries:


- Learned Concepts or functions:
- Tuple_data_type
- Enumerate_built_in_function
- strip_built_in_function
- todo: list_built_in_function
- todo: learn List Comprehension
- todo: learn set data type
- todo: iter built_int_function
- todo: next built_int_function

"""


def learn_tuple_data_type():
    """ Theory:
     - Tuple Değiştirlemez sıralı bir veri yapısıdır. Immutable olduğu için sonradan değiştirilemez
     -

    """

    #create tuple
    my_tuple_1 = (1, 2, 3)
    my_tuple_2 = 1, 2, 3,

    print(my_tuple_1, type(my_tuple_1))
    print(my_tuple_2, type(my_tuple_2))

def learn_enumerate_built_in_function():
    """
    - bir liste, tuple veya iterable nesnenin elemanlarını numaralandırarak döndüren fonksyiondur.
    - Enumarete() listedeki her öğeye otomatik bir indeks numarası ekler ve bu indeks ile ilgili öğeyi
    bir tuple içerisinde döndürür.
    - enumarete bize dönen objeyi listeye çevirdiğimizde gördüğümüz gibi bize bir
    [(0, 'apple'), (1, 'banana'), (2, 'melon')] tuple key, value pair dönüyor.

    :return:
    """
    my_fruits = ["apple", "banana", "melon"]
    my_returned_tuple_from_enumerate = enumerate(my_fruits)
    print(list(my_returned_tuple_from_enumerate))

    #use case 1
    for index, value in enumerate(my_fruits):
        print("index: ", index, " value: ", value)

def learn_strip_built_in_function():
    """
    -  bir string'in başındaki ve sonundaki boşlukları
     (whitespace) veya belirttiğimiz karakterleri temizler.
    :return:
    """
    text = " Mustafa Karacabey is engineer "
    new_text_without_whitespace = text.strip()
    print(text)
    print(new_text_without_whitespace)





if __name__ == "__main__":
    #learn_tuple_data_type()
    #learn_enumerate_built_in_function()
    learn_strip_built_in_function()