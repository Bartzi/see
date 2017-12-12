import argparse
import csv
import json
import os

from tqdm import tqdm


def extract_words_from_gt(gt_file, extracted_words):
    with open(gt_file) as gt:
        reader = csv.reader(gt, delimiter='\t')
        lines = [l for l in reader]

    for line in tqdm(lines):
        labels = line[1:]
        text_line = ''.join([chr(char_map[x]) for x in labels])
        words = text_line.strip(chr(char_map[args.blank_label])).split()

        for word in words:
            extracted_words.add(word)

    return extracted_words


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract all words used in fsns dataset")
    parser.add_argument("gt", help='path to fsns gt dir')
    parser.add_argument("dest", help='destination dict')
    parser.add_argument("char_map", help='path to char map')
    parser.add_argument('--blank-label', default='133', help='class number of blank label')

    args = parser.parse_args()

    with open(args.char_map) as the_char_map:
        char_map = json.load(the_char_map)
        reverse_char_map = {v: k for k, v in char_map.items()}

    gt_files = filter(lambda x: os.path.splitext(x)[-1] == '.csv', os.listdir(args.gt))

    words = set()
    for gt_file in gt_files:
        words = extract_words_from_gt(os.path.join(args.gt, gt_file), words)

    with open(args.dest, 'w') as destination:
        print(len(words), file=destination)
        for word in words:
            print(word, file=destination)
