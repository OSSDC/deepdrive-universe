class GTASetting(object):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __repr__(self):
        return str(type(self)) + '<{}={}>'.format(self.key, self.value)

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        return type(other) == type(self) and \
               other.amount == self.value

    def compile(self):
        return 'GameSettings', self.key, self.value

