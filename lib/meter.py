# ## Logger
class Meter(object):
    def __init__(self, **kwargs):
        self.reset()
        self.format(**kwargs)

    def reset(self):
        self.fmt = dict()
        self.values = dict()

    def format(self, **kwargs):
        for key, fmt_str in kwargs.items():
            self.fmt[key] = fmt_str

    def average_value(self):
        return {key: sum(vals) / len(vals) for key, vals in self.values.items()}

    def last_value(self):
        return {key:  vals[-1] for key, vals in self.values.items()}

    def update(self, **kwargs):
        for key, val in kwargs.items():
            self._update(key, val)

    def _create(self, key):
        if key not in self.fmt.keys():
            self.fmt[key] = ':5.3f'
        self.values[key] = []

    def _update(self, key, val):
        if key not in self.values.keys():
            self._create(key)
        self.values[key].append(val)

    def _str(self, key, val):
        return ('{key:s}: {val' + self.fmt[key] + '}').format(key=key, val=val)


class AverageMeter(Meter):
    def __init__(self, **kwargs):
        super(AverageMeter, self).__init__()

    def __str__(self):
        return ', '.join([
            self._str(key, val) for key, val in self.average_value().items()
        ])


class LastMeter(Meter):
    def __init__(self, **kwargs):
        super(LastMeter, self).__init__()

    def __str__(self):
        return ', '.join([
            self._str(key, val) for key, val in self.last_value().items()
        ])
