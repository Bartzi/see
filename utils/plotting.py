import argparse
import json

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


class LogPlotter(object):

    def __init__(self, log_file):
        self.log_file = log_file
        self.train_iterations = {}
        self.test_iterations = {}

    def parse_log_file(self, start=0, end=None):
        with open(self.log_file) as log_file:
            log_data = json.load(log_file)
            if end is None:
                end = log_data[-1]['iteration']

            events_to_plot = filter(lambda x: start <= x['iteration'] <= end, log_data)
            for event in events_to_plot:
                iteration = event.pop('iteration')
                self.train_iterations[iteration] = {
                    key.rsplit('/')[-1]: event[key] for key in
                    filter(lambda x: ('accuracy' in x or 'loss' in x) and 'validation' not in x, event)
                }

                test_keys = list(filter(lambda x: 'validation' in x, event))
                if len(test_keys) > 0:
                    self.test_iterations[iteration] = {
                        key.rsplit('/')[-1]: event[key] for key in
                        filter(lambda x: ('accuracy' in x or 'loss' in x), test_keys)
                    }

    def plot(self, start=0, end=None):
        self.parse_log_file(start=start, end=end)

        metrics_to_plot = sorted(next(iter(self.train_iterations.values())).keys(), key=lambda x: x.rsplit('_'))
        fig, axes = plt.subplots(len(metrics_to_plot), sharex=True)

        x_train = list(sorted(self.train_iterations.keys()))
        x_test = list(sorted(self.test_iterations.keys()))

        for metric, axe in zip(metrics_to_plot, axes):
            axe.plot(x_train, [self.train_iterations[iteration][metric] for iteration in x_train], 'r.-', label='train')
            axe.plot(x_test, [self.test_iterations[iteration][metric] for iteration in x_test], 'g.-', label='test')

            axe.set_title(metric)

            box = axe.get_position()
            axe.set_position([box.x0, box.y0, box.width * 0.9, box.height])

            axe.legend(bbox_to_anchor=(1, 0.5), loc='center left', fancybox=True, shadow=True)

        return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool to create plots of training')
    parser.add_argument("log_file", help="path to log file")
    parser.add_argument("-d", "--destination", dest='destination', help='where to save the resulting plot')
    parser.add_argument("-f", "--from", dest='start', default=0, type=int, help="start index from which you want to plot")
    parser.add_argument("-t", "--to", dest='end', type=int, help="index until which you want to plot (default: end)")

    args = parser.parse_args()

    plotter = LogPlotter(args.log_file)
    fig = plotter.plot(start=args.start, end=args.end)
    if args.destination is None:
        plt.show()
    else:
        fig.savefig(args.destination)