import argparse
import csv

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that removes a given prefix from the file names in the given csv file')
    parser.add_argument("csv_file", help="path to file where paths shall be changed")
    parser.add_argument("prefix", help='prefix to remove')

    args = parser.parse_args()

    with open(args.csv_file) as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        lines = [l for l in reader]

    for line in tqdm(lines[1:]):
        path = line[0]
        path = path[len(args.prefix):]
        line[0] = path

    with open(args.csv_file, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerows(lines)
