import argparse

import json

import csv

import itertools
import tqdm as tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that converts fsns gt to per word gt")
    parser.add_argument('fsns_gt', help='apth to original fsns gt file')
    parser.add_argument('char_map', help='path to fsns char map')
    parser.add_argument('destination', help='path to destination gt file')
    parser.add_argument('--max-words', type=int, default=6, help='max words per image')
    parser.add_argument('--max-chars', type=int, default=21, help='max characters per word')
    parser.add_argument('--blank-label', default='133', help='class number of blank label')
    parser.add_argument('--convert-to-single-line', action='store_true', default=False, help='convert gt to a single line gt')

    args = parser.parse_args()

    with open(args.char_map) as c_map:
        char_map = json.load(c_map)
        reverse_char_map = {v: k for k, v in char_map.items()}

    with open(args.fsns_gt) as fsns_gt:
        reader = csv.reader(fsns_gt, delimiter='\t')
        lines = [l for l in reader]

        text_lines = []
        for line in tqdm.tqdm(lines):
            text = ''.join(map(lambda x: chr(char_map[x]), line[1:]))
            words = text.split(chr(char_map[args.blank_label]))
            words = filter(lambda x: len(x) > 0, words)

            text_line = ' '.join(words)

            # pad resulting data with blank label
            text_line += ''.join(chr(char_map[args.blank_label]) * (37 - len(text_line)))
            text_line = [reverse_char_map[ord(character)] for character in text_line]

            text_lines.append([line[0], *text_line])

    with open(args.destination, 'w') as dest:
        writer = csv.writer(dest, delimiter='\t')
        writer.writerows(text_lines)
