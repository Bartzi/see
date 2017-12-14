import argparse
import os
import urllib.request


BASE_URL = "http://download.tensorflow.org/data/fsns-20160927/"

SETS = [
	('test', 0, 64),
	('train', 0, 512),
	('validation', 0, 64),
]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='tool that downloads fsns dataset')
	parser.add_argument("destination_dir", help='path to destination directory')

	args = parser.parse_args()

	for set_name, start_part, num_parts in SETS:
		dest_dir = os.path.join(args.destination_dir, set_name)
		os.makedirs(dest_dir, exist_ok=True)

		for part in range(start_part, num_parts):
			file_name = "{set_name}-{part:0>5}-of-{num_parts:0>5}".format(
				set_name=set_name,
				part=part,
				num_parts=num_parts,
			)

			url = "{base}{set_name}/{file_name}".format(
				base=BASE_URL,
				set_name=set_name,
				file_name=file_name,
			)

			print("downloading {}".format(file_name))
			with urllib.request.urlopen(url) as url_data, open(os.path.join(dest_dir, file_name), 'wb') as f:
				file_size = int(url_data.info()['Content-Length'])

				downloaded = 0
				block_size = 8192
				while True:
					buffer = url_data.read(block_size)
					if not buffer:
						break

					downloaded += len(buffer)
					f.write(buffer)
					print("Got: {:>10} of {:>10} bytes".format(downloaded, file_size), end='\r')

			print("{}".format(" " * 100), end='\r')