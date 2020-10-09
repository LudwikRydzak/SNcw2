import random
import numpy as np

class Adaline:
    def __init__(self, _entry_count, _learning_factor):
        self.entry_values = []
        self.entry_weights = []
        self.entry_values.append(1)
        self.bias = 1
        self.entry_weights.append(round((random.random() / 10), 4))

        for i in range(_entry_count):
            self.entry_values.append(0)
            self.entry_weights.append(round((random.random() / 10), 4))
        self.learning_factor = _learning_factor

    def bipolar_function(self, _sum):
        return 1 if _sum > 0 else -1


    def unipolar_function(self, _sum):
        return 1 if _sum > 0 else 0


    def output(self, _sum, _uni0_bi1):
        if _uni0_bi1 == 0:
            return self.unipolar_function(_sum)
        if _uni0_bi1 == 1:
            return self.bipolar_function(_sum)


    def entry_function(self, _entry_values, _uni0_bi1):
        for i in range(len(_entry_values)):
            self.entry_values[i + 1] = _entry_values[i]

    def single_error(self, _label):
        sum =(_label - np.dot(self.entry_values, self.entry_weights)) ** 2
        return sum

    def error(self, _labels, _input_sets):
        sum = 0
        for i in range(len(_labels)):
            self.entry_function(_input_sets[i])
            sum += self.single_error(_labels[i])