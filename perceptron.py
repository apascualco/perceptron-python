import numpy


class Perceptron(object):

    def __init__(self, learning_rate=0.01, iterations=50, random_state=1):
        self.weight = None
        self.prediction_error = []
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.random_state = random_state

    def generate_random_weight(self, shape):
        random_state = numpy.random.RandomState(self.random_state)
        return random_state.normal(loc=0.0, scale=0.01, size=1 + shape)

    def training(self, x, y):
        self.weight = self.generate_random_weight(x.shape[1])
        for _ in range(self.iterations):
            prediction_fail = 0
            for entry, target in zip(x, y):
                update = self.learning_rate * (target - self.threshold_function(entry))
                if update != 0.0:
                    self.weight[1:] += update * entry
                    self.weight[0] += update
                    prediction_fail += 1
            self.prediction_error.append(prediction_fail)
        return self

    def scalar(self, x):
        return numpy.dot(x, self.weight[1:]) + self.weight[0]

    def threshold_function(self, entry):
        return numpy.where(self.scalar(entry) >= 0.0, 1, -1)
