from Generator import *
from Adaline import *


def test():
    ranges = [[-1, 1], [-0.9, 0.9], [-0.8, 0.8], [-0.7, 0.7], [-0.6, 0.6], [-0.5, 0.5], [-0.4, 0.4], [-0.3, 0.3],
              [-0.2, 0.2], [-0.15, 0.15], [-0.1, 0.1], [-0.05, 0.05], [-0.01, 0.01]]
    max_error = [3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01,]
    learning_rate = [3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005,
                     0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    learning_count = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    max_epochs = [50, 100, 200, 400, 800]
    testing_count = 1000

    with open('wynikiBadaniaAdaline.txt', 'w') as file:
        file.writelines(
            'learning count; testing count; learning rate; max_epochs, range; max_error; epochs of learning; percentage of good answers\n')
        testing_set = generate(testing_count)
        for count in learning_count:
            print('|')
            learning_set = generate(count)
            for range in ranges:
                print('o')
                for epochs in max_epochs:
                    print('.')
                    for rate in learning_rate:
                        for error in max_error:
                            adaline = Adaline(2, rate, epochs, range[0], range[1])
                            epochs_of_learning = adaline.learn(learning_set, error)
                            percentage = adaline.test_adaline(testing_set)
                            file.writelines(
                                f'{count};{testing_count};{rate};{epochs};{range};{error};{epochs_of_learning};{percentage}\n')
