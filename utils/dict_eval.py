import argparse
import csv

import hunspell
from collections import Counter
from itertools import zip_longest

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tool that does extra dictionary correction on top of given predictions, including gt")
    parser.add_argument("evaluation_result", help="path to eval result file")
    parser.add_argument("dictionary", help="path to dictionary file")
    parser.add_argument("affix", help="path to affix file")
    parser.add_argument("--extra-dict", help="extra file containing words to add to dict")

    args = parser.parse_args()

    with open(args.evaluation_result) as eval_result_file:
        reader = csv.reader(eval_result_file)
        lines = [l for l in reader]

    hobj = hunspell.HunSpell(args.dictionary, args.affix)

    if args.extra_dict:
        with open(args.extra_dict) as extra_dict:
            for line in extra_dict:
                hobj.add(line)

    num_word_x_correct = Counter()
    num_word_x = Counter()
    correct_words = 0
    num_words = 0
    num_correct_lines = 0
    num_lines = 0

    for prediction, gt in tqdm(lines):
        corrected_words = []
        for word in prediction.split():
            corrected_word = word
            if not hobj.spell(word):
                suggestions = hobj.suggest(word)
                if len(suggestions) > 0:
                    corrected_word = suggestions[0]

            corrected_words.append(corrected_word)

        line_correct = True
        for i, (corrected_word, gt_word) in enumerate(zip_longest(corrected_words, gt.split(), fillvalue=''), start=1):
            if corrected_word.lower() == gt_word.lower():
                correct_words += 1
                num_word_x_correct[i] += 1
            else:
                line_correct = False

            num_words += 1
            num_word_x[i] += 1
        if line_correct:
            num_correct_lines += 1
        num_lines += 1

    print("Sequence Accuracy: {}".format(num_correct_lines / num_lines))
    print("Word Accuracy: {}".format(correct_words / num_words))
    print("Single word accuracies:")
    for i in range(1, len(num_word_x) + 1):
        print("Accuracy for Word {}: {}".format(i, num_word_x_correct[i] / num_word_x[i]))
