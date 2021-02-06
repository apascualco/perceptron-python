import matplotlib.pyplot as pyplot
import pandas
import numpy
import perceptron as p


def show_data_in_graph(entries):
    pyplot.scatter(entries[:50, 0], entries[:50, 1], color='red', marker='o', label='setosa')
    pyplot.scatter(entries[50:100, 0], entries[50:100, 1], color='blue', marker='x', label='versicolor')
    pyplot.xlabel('sepal [cm]')
    pyplot.ylabel('petal [cm]')
    pyplot.legend(loc='lower right')
    pyplot.show()


def show_training_data(perceptron):
    pyplot.plot(range(1, len(perceptron.prediction_error) + 1), perceptron.prediction_error, marker='o')
    pyplot.xlabel('Iterations')
    pyplot.ylabel('Number of prediction errors')
    pyplot.show()


if __name__ == '__main__':
    data = pandas.read_csv('iris.data', header=None)
    entries = data.iloc[0:100, [0, 2]].values

    show_data_in_graph(entries)

    target = data.iloc[0:100, 4].values
    target_weight = numpy.where(target == 'Iris-setosa', -1, 1)

    perceptron = p.Perceptron(learning_rate=0.1, iterations=10)
    perceptron.training(entries, target_weight)
    show_training_data(perceptron)
