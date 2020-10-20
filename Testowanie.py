from Generator import *
from Adaline import *
import matplotlib.pyplot as plt


def test():
    ranges = [[-100, 100],[-50, 50],[-20, 20],[-10, 10],[-5, 5],[-1, 1], [-0.9, 0.9], [-0.8, 0.8], [-0.7, 0.7], [-0.6, 0.6], [-0.5, 0.5], [-0.4, 0.4], [-0.3, 0.3],
              [-0.2, 0.2], [-0.15, 0.15], [-0.1, 0.1], [-0.05, 0.05], [-0.01, 0.01]]
    max_errors = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, ]
    learning_rate = [3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005,
                     0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    learning_count = 100
    max_epochs = 100
    testing_count = 25
    wagi(learning_count, testing_count, learning_rate, max_epochs, max_errors, ranges)
    # wsp_uczenia(learning_count, testing_count, learning_rate, max_epochs, max_errors, ranges)
    # max_error(learning_count, testing_count, learning_rate, max_epochs, max_errors, ranges)

def wagi(learning_count, testing_count, learning_rate, max_epochs, max_errors, ranges):
    with open('badaniaWeightRange.txt', 'w') as file:
        file.writelines(
            'learning count; testing count; learning rate; max_epochs; max_error; range;max epochs of learning; mean epochs of learning; min epochs of learning; max percentage of good answers; mean percentage of good answers; min percentage of good answers\n')
        testing_set = generate(testing_count)
        learning_set = generate(learning_count)
        epochs_min_set = []
        percentage_min_set = []
        epochs_mean_set = []
        percentage_mean_set = []
        epochs_max_set = []
        percentage_max_set = []
        for weights_range in ranges:
            epochs_mean = 0
            percentage_mean = 0
            epochs_max = 0
            epochs_min = 501
            percentage_max = 0
            percentage_min = 101
            for i in range(10):
                print(i)
                adaline = Adaline(2, 0.1, max_epochs, weights_range[0], weights_range[1])
                epochs_of_learning = adaline.learn(learning_set, 0.3)
                percentage = adaline.test_adaline(testing_set)
                epochs_mean += epochs_of_learning
                percentage_mean += percentage
                if epochs_of_learning > epochs_max:
                    epochs_max = epochs_of_learning
                if epochs_of_learning < epochs_min:
                    epochs_min = epochs_of_learning
                if percentage > percentage_max:
                    percentage_max = percentage
                if percentage < percentage_min:
                    percentage_min = percentage
            epochs_mean /= 10
            percentage_mean /= 10
            epochs_min_set.append(epochs_min)
            percentage_min_set.append(percentage_min)
            epochs_mean_set.append(epochs_mean)
            percentage_mean_set.append(percentage_mean)
            epochs_max_set.append(epochs_max)
            percentage_max_set.append(percentage_max)
            file.writelines(
                f'{learning_count};{testing_count};{0.1};{max_epochs};{0.3};{weights_range};{epochs_max};{epochs_mean};{epochs_min};{percentage_max};{percentage_mean};{percentage_min}\n')
        show_range_epochs(ranges, epochs_min_set, epochs_mean_set, epochs_max_set)
        show_range_percentages(ranges, percentage_min_set, percentage_mean_set, percentage_max_set)


def wsp_uczenia(learning_count, testing_count, learning_rate, max_epochs, max_errors, ranges):
    with open('badaniaLearningRate.txt', 'w') as file:
        file.writelines(
            'learning count; testing count; learning rate; max_epochs; max_error; range;max epochs of learning; mean epochs of learning; min epochs of learning; max percentage of good answers; mean percentage of good answers; min percentage of good answers\n')
        testing_set = generate(testing_count)
        learning_set = generate(learning_count)
        epochs_min_set = []
        percentage_min_set = []
        epochs_mean_set = []
        percentage_mean_set = []
        epochs_max_set = []
        percentage_max_set = []
        for rate in learning_rate:
            epochs_mean = 0
            percentage_mean = 0
            epochs_max = 0
            epochs_min = 501
            percentage_max = 0
            percentage_min = 101
            for i in range(10):
                print(i)
                adaline = Adaline(2, rate, max_epochs, -1, 1)
                epochs_of_learning = adaline.learn(learning_set, 0.3)
                percentage = adaline.test_adaline(testing_set)
                epochs_mean += epochs_of_learning
                percentage_mean += percentage
                if epochs_of_learning > epochs_max:
                    epochs_max = epochs_of_learning
                if epochs_of_learning < epochs_min:
                    epochs_min = epochs_of_learning
                if percentage > percentage_max:
                    percentage_max = percentage
                if percentage < percentage_min:
                    percentage_min = percentage
            epochs_mean /= 10
            percentage_mean /= 10
            epochs_min_set.append(epochs_min)
            percentage_min_set.append(percentage_min)
            epochs_mean_set.append(epochs_mean)
            percentage_mean_set.append(percentage_mean)
            epochs_max_set.append(epochs_max)
            percentage_max_set.append(percentage_max)
            file.writelines(
                f'{learning_count};{testing_count};{rate};{max_epochs};{0.3};[-1, 1];{epochs_max};{epochs_mean};{epochs_min};{percentage_max};{percentage_mean};{percentage_min}\n')
        show_learning_epochs(learning_rate, epochs_min_set, epochs_mean_set, epochs_max_set)
        show_learning_percentages(learning_rate, percentage_min_set, percentage_mean_set, percentage_max_set)


def max_error(learning_count, testing_count, learning_rate, max_epochs, max_errors, ranges):
    with open('badaniaMaxError.txt', 'w') as file:
        file.writelines(
            'learning count; testing count; learning rate; max_epochs; max_error; range;max epochs of learning; mean epochs of learning; min epochs of learning; max percentage of good answers; mean percentage of good answers; min percentage of good answers\n')
        testing_set = generate(testing_count)
        learning_set = generate(learning_count)
        epochs_min_set = []
        percentage_min_set = []
        epochs_mean_set = []
        percentage_mean_set = []
        epochs_max_set = []
        percentage_max_set = []
        for max_error in max_errors:
            epochs_mean = 0
            percentage_mean = 0
            epochs_max = 0
            epochs_min = 501
            percentage_max = 0
            percentage_min = 101
            for i in range(10):
                print(i)
                adaline = Adaline(2, 0.1, max_epochs, -1, 1)
                epochs_of_learning = adaline.learn(learning_set, max_error)
                percentage = adaline.test_adaline(testing_set)
                epochs_mean += epochs_of_learning
                percentage_mean += percentage
                if epochs_of_learning > epochs_max:
                    epochs_max = epochs_of_learning
                if epochs_of_learning < epochs_min:
                    epochs_min = epochs_of_learning
                if percentage > percentage_max:
                    percentage_max = percentage
                if percentage < percentage_min:
                    percentage_min = percentage
            epochs_mean /= 10
            percentage_mean /= 10
            epochs_min_set.append(epochs_min)
            percentage_min_set.append(percentage_min)
            epochs_mean_set.append(epochs_mean)
            percentage_mean_set.append(percentage_mean)
            epochs_max_set.append(epochs_max)
            percentage_max_set.append(percentage_max)
            file.writelines(
                f'{learning_count};{testing_count};{0.1};{max_epochs};{max_error};[-1, 1];{epochs_max};{epochs_mean};{epochs_min};{percentage_max};{percentage_mean};{percentage_min}\n')
        show_error_epochs(max_errors, epochs_min_set, epochs_mean_set, epochs_max_set)
        show_error_percentages(max_errors, percentage_min_set, percentage_mean_set, percentage_max_set)


def show_learning_epochs(x, ymin, ymean, ymax):
    plt.title('Badanie wpływu współczynnika uczenia na szybkość uczenia w epokach')
    plt.xlabel('współczynnik uczenia')
    plt.ylabel('epoki')
    plt.plot(x, ymin, 'r+', label='minimalna liczba epok')
    plt.plot(x, ymean, 'bo', label='średnia liczba epok')
    plt.plot(x, ymax, 'g+', label='maksymalna liczba epok')
    plt.legend()
    plt.show()


def show_learning_percentages(x, ymin, ymean, ymax):
    plt.title('Badanie wpływu współczynnika uczenia na jakość uczenia w procentach')
    plt.xlabel('współczynnik uczenia')
    plt.ylabel('% poprawnych odpowiedzi')
    plt.plot(x, ymin, 'r+', label='minimalny uzyskany procent')
    plt.plot(x, ymean, 'bo', label='średni uzyskany procent')
    plt.plot(x, ymax, 'g+', label='maksymalny uzyskany procent')
    plt.legend()
    plt.show()


def show_range_epochs(x, ymin, ymean, ymax):
    plt.title('Badanie wpływu zakresu inicjowania wag na szybkość uczenia w epokach')
    plt.xlabel('zakres wag')
    plt.ylabel('epoki')
    plt.plot(x, ymin, 'r+', label='minimalna liczba epok')
    plt.plot(x, ymean, 'bo', label='średnia liczba epok')
    plt.plot(x, ymax, 'g+', label='maksymalna liczba epok')
    plt.legend()
    plt.show()


def show_range_percentages(x, ymin, ymean, ymax):
    plt.title('Badanie wpływu zakresu inicjowania wag na jakość uczenia w procentach')
    plt.xlabel('zakres wag')
    plt.ylabel('% poprawnych odpowiedzi')
    plt.plot(x, ymin, 'r+', label='minimalny uzyskany procent')
    plt.plot(x, ymean, 'bo', label='średni uzyskany procent')
    plt.plot(x, ymax, 'g+', label='maksymalny uzyskany procent')
    plt.legend()
    plt.show()


def show_error_epochs(x, ymin, ymean, ymax):
    plt.title('Badanie wpływu maksymalnego dopuszczalnego błędu na szybkość uczenia w epokach')
    plt.xlabel('zakres wag')
    plt.ylabel('epoki')
    plt.plot(x, ymin, 'r+', label='minimalna liczba epok')
    plt.plot(x, ymean, 'bo', label='średnia liczba epok')
    plt.plot(x, ymax, 'g+', label='maksymalna liczba epok')
    plt.legend()
    plt.show()


def show_error_percentages(x, ymin, ymean, ymax):
    plt.title('Badanie wpływu maksymalnego dopuszczalnego błędu na jakość uczenia w procentach')
    plt.xlabel('zakres wag')
    plt.ylabel('% poprawnych odpowiedzi')
    plt.plot(x, ymin, 'r+', label='minimalny uzyskany procent')
    plt.plot(x, ymean, 'bo', label='średni uzyskany procent')
    plt.plot(x, ymax, 'g+', label='maksymalny uzyskany procent')
    plt.legend()
    plt.show()
