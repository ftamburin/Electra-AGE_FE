#! /usr/bin/env python3
"""Generate lexicon files from Rofames train.frames splits."""

import sys
import codecs

def generate_lexicon(input_file_path, output_file_path):
    lexicon_items = set()
    with codecs.open(input_file_path, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            tokens = line.split('\t')
            lexicon_items.add('{}\t{}'.format(tokens[3], tokens[4]))
    with codecs.open(output_file_path, 'w', encoding='utf-8') as output_stream:
        for item in sorted(lexicon_items):
            print(item, file=output_stream)


if __name__ == '__main__':
    INPUT_FILE_PATH = sys.argv[1]
    OUTPUT_FILE_PATH = sys.argv[2]
    generate_lexicon(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
