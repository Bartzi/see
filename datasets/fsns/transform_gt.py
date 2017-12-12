import argparse
import csv
import json

import itertools
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that converts fsns gt to per word gt")
    parser.add_argument('fsns_gt', help='path to original fsns gt file')
    parser.add_argument('char_map', help='path to fsns char map')
    parser.add_argument('destination', help='path to destination gt file')
    parser.add_argument('--max-words', type=int, default=6, help='max words per image')
    parser.add_argument('--min-words', type=int, default=1, help='min words per image')
    parser.add_argument('--max-chars', type=int, default=21, help='max characters per word')
    parser.add_argument('--blank-label', default='133', help='class number of blank label')
    parser.add_argument('--word-gt', action='store_true', default=False, help='input gt is word level gt')

    args = parser.parse_args()

    with open(args.char_map) as c_map:
        char_map = json.load(c_map)
        reverse_char_map = {v: k for k, v in char_map.items()}

    with open(args.fsns_gt) as fsns_gt:
        reader = csv.reader(fsns_gt, delimiter='\t')
        lines = [l for l in reader]

    text_lines = []
    for line in tqdm(lines):
        text = ''.join(map(lambda x: chr(char_map[x]), line[1:]))
        if args.word_gt:
            text = text.split(chr(char_map[args.blank_label]))
            text = filter(lambda x: len(x) > 0, text)
        else:
            text = text.strip(chr(char_map[args.blank_label]))
            text = text.split()

        words = []
        for t in text:
            t = list(map(lambda x: reverse_char_map[ord(x)], t))
            t.extend([args.blank_label] * (args.max_chars - len(t)))
            words.append(t)

        if len(words) > args.max_words or len(words) < args.min_words:
            continue
        elif any([len(word) > args.max_chars for word in words]):
            continue

        words.extend([[args.blank_label] * args.max_chars for _ in range(args.max_words - len(words))])

        text_lines.append([line[0]] + list(itertools.chain(*words)))

    with open(args.destination, 'w') as dest:
        writer = csv.writer(dest, delimiter='\t')
        writer.writerow([args.max_words, args.max_chars])
        writer.writerows(text_lines)
