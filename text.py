import re
from params import MAX_LEN_INPUT, MAX_LEN_TARGET


p1 = re.compile(r'[^0-9a-zA-Z :;]')
p2 = re.compile(r' +')


def clean(text):
    return p2.subn(' ', p1.subn('', text)[0])[0]


def prepare_text(text_path):

    input_texts = []
    target_texts = []

    with open(text_path) as file:
        lines = file.readlines()

    for line in lines:
        target_text, input_text = map(clean, line.rstrip().split('\t'))

        if (len(input_text.split()) <= MAX_LEN_INPUT) and (len(target_text.split()) <= MAX_LEN_TARGET - 1):
            input_texts.append(input_text)
            target_texts.append(target_text)
    
    string = f'Load {len(input_texts)} texts({len(input_texts) / len(lines) * 100:.1f}%) from {text_path}'
    print(string)
    
    return input_texts, target_texts
