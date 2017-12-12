import threading

from cmd import Cmd


class InteractiveTrain(Cmd):

    prompt = ''

    def __init__(self, *args, **kwargs):
        self.bbox_plotter = kwargs.pop('bbox_plotter')
        self.curriculum = kwargs.pop('curriculum', None)
        self.lr_shifter = kwargs.pop('lr_shifter', None)

        super().__init__(*args, **kwargs)

    def do_enablebboxvis(self, arg):
        """Enable sending of bboxes to remote host"""
        self.bbox_plotter.send_bboxes = True

    def do_increasedifficulty(self, arg):
        """Increase dfficulty of learning curriculum"""
        if self.curriculum is not None:
            self.curriculum.force_enlarge_dataset = True

    def do_shiftlr(self, arg):
        if self.lr_shifter is not None:
            self.lr_shifter.force_shift = True

    def do_quit(self, arg):
        return True

    def do_echo(self, arg):
        print(arg)


def open_interactive_prompt(*args, **kwargs):
    cmd_interface = InteractiveTrain(*args, **kwargs)

    thread = threading.Thread(target=lambda: cmd_interface.cmdloop())
    thread.daemon = True
    thread.start()
