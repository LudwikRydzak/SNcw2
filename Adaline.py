import random
import sys

import numpy as np

class Adaline:
    def __init__(self, _entry_count, _learning_factor, _max_epochs, _weight_range_begin, _weight_range_stop):
        self.entry_values = []
        self.entry_weights = []
        self.max_epochs = _max_epochs
        self.entry_values.append(1)
        self.entry_weights.append(random.random() * (_weight_range_stop-_weight_range_begin) + _weight_range_begin)
        self.learning_factor = _learning_factor

        for i in range(_entry_count):
            self.entry_values.append(0)
            self.entry_weights.append(random.random() * (_weight_range_stop-_weight_range_begin) + _weight_range_begin)


    def bipolar_function(self, _sum):
        return 1 if _sum > 0 else -1

    def sum(self):
        return np.dot(self.entry_values, self.entry_weights)

    def entry_function(self, _entry_values):
        for i in range(len(_entry_values)):
            self.entry_values[i+1] = _entry_values[i]

    def single_error(self, _label):
        sum = (_label - np.dot(self.entry_values, self.entry_weights)) ** 2
        return sum

    def error(self, _labels, _entry_values):
        sum = 0
        for i in range(len(_labels)):
            self.entry_function(_entry_values[i])
            sum += self.single_error(_labels[i])
        return sum / (len(_entry_values))

    def change_weights(self, _label):
        for i in range(len(self.entry_weights)):

            self.entry_weights[i] = self.entry_weights[i] + self.learning_factor * (_label- np.dot(self.entry_values, self.entry_weights)) * self.entry_values[i]


    def learn(self, _training_set, _max_error):
        labels=[]
        sets =[]
        for i in range(len(_training_set)):
            set = _training_set[i]
            labels.append(set[0])
            sets.append(set[1:])
        epoch = 0;
        error = 9999999
        while ((error > _max_error) and (epoch < self.max_epochs)):
            error = self.error(labels, sets)
            for i in range(len(sets)):
                self.entry_function(sets[i])
                self.change_weights(labels[i])
            epoch += 1
        # print (f'nauczony po {epoch} epokach')
        # print(f'wyuczone wagi: {self.entry_weights}')
        return epoch

    def predict(self, _set):
        self.entry_function(_set)
        return self.bipolar_function(self.sum())


    def test_adaline(self, _test_sets):
        correct_answers = 0
        wrong_answers = 0
        for i in range(len(_test_sets)):
            test_set = _test_sets[i]
            label, set = test_set[0], test_set[1:]
            output = self.predict(set)
            if(output == label):
                correct_answers += 1
            else:
                wrong_answers += 1
        percent = round(correct_answers/(correct_answers+wrong_answers), 4) *100
        return percent