from chainer.optimizer import GradientMethod


class MultiNetOptimizer(GradientMethod):

    def __getattribute__(self, item):
        """Proxy all calls that can not be handled by this class to base optimizer"""
        try:
            v = object.__getattribute__(self, item)
        except AttributeError:
            v = getattr(object.__getattribute__(self, 'base_optimizer'), item)
        return v

    def __init__(self, base_optimizer, optimize_all_interval=10):
        self.base_optimizer = base_optimizer
        self.optimize_all_interval = optimize_all_interval
        super().__init__()
        self.extra_links = None
        self.hyperparam = self.base_optimizer.hyperparam

    def setup(self, link, extra_links=()):
        self.extra_links = extra_links
        super().setup(link)

    def update(self, lossfun=None, *args, **kwargs):
        """Updates parameters based on a loss function or computed gradients.
            This method runs in two ways.
            - If ``lossfun`` is given, then it is used as a loss function to
              compute gradients.
            - Otherwise, this method assumes that the gradients are already
              computed.
            In both cases, the computed gradients are used to update parameters.
            The actual update routines are defined by the update rule of each
            parameter.
        """
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwargs)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            loss.backward()
            del loss

        self.reallocate_cleared_grads()

        self.call_hooks()

        self.t += 1
        self.base_optimizer.t = self.t

        if self.t % self.optimize_all_interval == 0:
            # update all params
            for param in self.target.params():
                param.update()
        else:
            # only update params in extra links
            for link in self.extra_links:
                for param in link.params():
                    param.update()

    def create_update_rule(self):
        return self.base_optimizer.create_update_rule()
