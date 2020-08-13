import copy


def linear(progress):  # progress is between 0 and 1
    return progress


class Variable:
    def __init__(self, start=1, end=0, nb_steps=1, evolution=linear):
        self.start = start
        self.end = end
        self.nb_steps = nb_steps
        self.evolution = evolution

        self.iteration = 0
        self.difference = self.start - self.end
        self.current = self.start

    def get(self):
        return self.current

    def execute(self):
        if self.iteration < self.nb_steps:
            self.iteration += 1
        self.current = self.start - self.difference * self.evolution(self.iteration/self.nb_steps)


class Parameters:
    def __init__(self, dictionary=None):
        if dictionary is None:
            dictionary = {}
        self.data = dictionary

    def __getitem__(self, item):
        if item in self.data:
            return copy.deepcopy(self.data[item])
        return None

    def __setitem__(self, key, value):
        self.data[key] = copy.deepcopy(value)
