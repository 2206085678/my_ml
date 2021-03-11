# @Time    : 2021/3/7 21:36 
# @Author  : 孙北晨 
# @Version : V 0.1
# @Int     : 感知机...

import numpy
class Perceptron:
    def __init__(self, f, input_num):
        self.f = f
        self.weights = numpy.ones(input_num)
        self.b = 1
        self.input_num = input_num

    def predict(self, input_vec):
        return self.f(numpy.dot(input_vec, self.weights) + self.b)

    def train(self, input_vecs, lables, rate):
        for i in range(len(input_vecs)):
            self.train_helper(input_vecs[i], lables[i], rate)

    def train_helper(self, input_vec, lable, rate):
        y_hat = self.predict(input_vec)
        self.weights = self.weights + rate * (lable - y_hat) * input_vec
        print(input_vec)
        print(lable)
        print("*************")
        self.b += rate * (lable - y_hat)

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.b)

    def print_loss(self, fun):
        return


def f(x):

    return 1 if x > 0 else 0


if __name__ == '__main__':
    p = Perceptron(f, 2)
    input_vecs = numpy.array([[1., 1.], [0., 0.], [1., 0.], [0., 1.]])
    lables = numpy.array([1., 0., 0., 0.])
    for i in range(10):
        p.train(input_vecs, lables, rate=0.1)
    print(p)
    print(p.predict([1,1]))