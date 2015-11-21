import random
import math
from functions import inner_product, sigmoid, diff_sigmoid

class input_layer:

    def __init__(self, filename):
        fp = open(filename)
        lines = fp.readlines()
        fp.close()
        self.vector = []
        self.tk = []
        for each in lines:
            temp = each.split('\n')[0].split(',')
            if (temp[-1] == '7') or (temp[-1] == '0'):
                newarr = [float(every) for every in temp]
                self.vector.append(newarr)
                self.vector[-1].insert(0, 1)

        columnmean = [0.0 for i in xrange(0, len(self.vector))]
        for j in xrange(0, len(self.vector)):
            for i in xrange(0, len(self.vector[0])):
                columnmean[i] += self.vector[j][i]

        for i in xrange(0, len(columnmean)):
            columnmean[i] /= len(self.vector)

        for each in self.vector:
            if each[-1] == 0:
                self.tk.append([1.0, 0.0])
            else:
                self.tk.append([0.0, 1.0])

        standarddev = [0.0 for i in xrange(0, len(self.vector))]
        for j in xrange(0, len(self.vector)):
            for i in xrange(0, len(self.vector[0])):
                standarddev[i] += (pow(self.vector[j][i] - columnmean[i], 2))

        for i in xrange(0, len(standarddev)):
            standarddev[i] /= len(self.vector)

        for i in xrange(0, len(self.vector)):
            for j in xrange(1, len(self.vector[0]) - 1):
                if standarddev[j] != 0:
                    self.vector[i][j] = (self.vector[i][j] - columnmean[j]) / standarddev[j]

    def get_vectors(self):
        return self.vector

    def get_tk(self):
        return self.tk



class hidden_layer:

    def __init__(self, num, veclen):
        self.output = []
        self.net = []
        self.weight = []
        self.delta_weight = []
        self.veclen = veclen
        self.num = num
        self.setout = []
    
        init_w = math.sqrt(veclen)
        for i in xrange(0, num):
            arr = [random.uniform(-1.0 / init_w, 1.0 / init_w) for j in xrange(0, veclen)]
            self.weight.append(arr)
            self.net.append(0.0)
            self.output.append(0.0)
            self.setout.append(0.0)

    def set_activation(self, i, vector):
        self.net[i] = inner_product(self.weight[i], vector)
        self.output[i] = sigmoid(self.net[i], 1.0)
        if self.output[i] > 0.5:
            self.setout[i] = 1.0
        else:
            self.setout[i] = 0.0

    def reset_delta_weight(self):
        self.delta_weight = []
        for i in xrange(0, self.num):
            arr = [0.0 for j in xrange(0, self.veclen)]
            self.delta_weight.append(arr)

    def set_delta_weight(self, i, j, val):
        self.delta_weight[i][j] += val

    def set_weight(self):
        for i in xrange(0, self.num):
            for j in xrange(0, self.veclen):
                self.weight[i][j] += self.delta_weight[i][j]

    def get_weight(self):
        return self.weight

    def get_net(self):
        return self.net

    def get_output(self):
        return self.output

    def get_set_out(self):
        return self.setout

class output_layer:

    def __init__(self, num, veclen):
        self.output = []
        self.net = []
        self.weight = []
        self.num = num
        self.veclen = veclen
        self.delta_weight = []
        self.setout = []

        init_w = math.sqrt(veclen)
        for i in xrange(0, num):
            arr = [random.uniform(-1.0 / init_w, 1.0 / init_w) for i in xrange(0, veclen)]
            self.weight.append(arr)
            self.net.append(0.0)
            self.output.append(0.0)
            self.setout.append(0.0)

    def set_activation(self, i, vector):
        self.net[i] = inner_product(self.weight[i], vector)
        self.output[i] = sigmoid(self.net[i], 1.0)
        if self.output[i] > 0.5:
            self.setout[i] = 1.0
        else:
            self.setout[i] = 0.0

    def reset_delta_weight(self):
        self.delta_weight = []
        for i in xrange(0, self.num):
            arr = [0.0 for j in xrange(0, self.veclen)]
            self.delta_weight.append(arr)

    def set_delta_weight(self, i, j, val):
        self.delta_weight[i][j] += val

    def set_weight(self):
        for i in xrange(0, self.num):
            for j in xrange(0, self.veclen):
                self.weight[i][j] += self.delta_weight[i][j]

    def get_weight(self):
        return self.weight

    def get_net(self):
        return self.net

    def get_output(self):
        return self.output

    def get_delta(self):
        return self.delta_weight

    def get_set_out(self):
        return self.setout

    #def putit(self):
    #    self.setout[0] = 0.0
    #    self.setout[1] = 1.0
    #    if self.output[0] > self.output[1]:
    #        self.setout[0] = 1.0
   #         self.setout[1] = 0.0


def test():
    x = input_layer('optdigits.tra')
    vec = x.get_vectors()
    print vec[0]
    print vec[1]
    y = hidden_layer(8, 64 + 1)
    y.reset_delta_weight()
    print 'len is ', len(y.get_weight())
    print 'length of ', len(y.get_weight()[5])
    print y.get_weight()[1][1]
    y.set_delta_weight(1, 1, 0.5)
    y.set_weight()
    print y.get_weight()[1][1]
    print 'net ', y.get_net()[1]
    print 'output ', y.get_output()[1]
    y.set_activation(1, vec[1])
    print 'net ', y.get_net()[1]
    print 'output ', y.get_output()[1]

    print "next test"
    y = output_layer(8, 64)
    y.reset_delta_weight()
    print 'len is ', len(y.get_weight())
    print 'length of ', len(y.get_weight()[5])
    print y.get_weight()[1][1]
    y.set_delta_weight(1, 1, 0.5)
    y.set_weight()
    print y.get_weight()[1][1]
    print 'net ', y.get_net()[1]
    print 'output ', y.get_output()[1]
    y.set_activation(1, vec[1])
    print 'net ', y.get_net()[1]
    print 'output ', y.get_output()[1]

