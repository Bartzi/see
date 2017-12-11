import os
import shutil

from chainer.training.extensions import LogReport


class Logger(LogReport):

    def __init__(self, model_files, log_dir, keys=None, trigger=(1, 'epoch'), postprocess=None, log_name='log'):
        super(Logger, self).__init__(keys=keys, trigger=trigger, postprocess=postprocess, log_name=log_name)
        self.backup_model(model_files, log_dir)

    def backup_model(self, model_files, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        for model_file in model_files:
            shutil.copy(model_file, log_dir)
