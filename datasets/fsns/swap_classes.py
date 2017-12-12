import argparse
import csv

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='swap to classes in gt')
    parser.add_argument('gt_file', help='path to gt file where class labels shall be swapped')
    parser.add_argument('destination', help='path to file with swapped labels')
    parser.add_argument('class1', help='first label to swap')
    parser.add_argument('class2', help='second label to swap')

    args = parser.parse_args()

    lines = []
    with open(args.gt_file) as gt_file:
        reader = csv.reader(gt_file, delimiter='\t')

        for line in tqdm(reader):
            new_line = []
            for label in line:
                if label == args.class1:
                    new_line.append(args.class2)
                elif label == args.class2:
                    new_line.append(args.class1)
                else:
                    new_line.append(label)
            lines.append(new_line)

    with open(args.destination, 'w') as destination:
        writer = csv.writer(destination, delimiter='\t')
        writer.writerows(lines)
