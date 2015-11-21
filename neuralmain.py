from layers import *
from functions import *
import math

def feed_forward(inp, hid, out, no_hid, no_out):
    for i in xrange(0, no_hid):
        hid.set_activation(i, inp)

    send = hid.get_output()

    for i in xrange(0, no_out):
        out.set_activation(i, send)

    #out.putit()


def back_propagation(inp, hid, out, no_hid, no_out, tvec):
    eta = 0.1
    outvec = out.get_output()
    outvec2 = out.get_set_out()
    outnet = out.get_net()
    hidout = hid.get_output()
    hidnet = hid.get_net()
    deltak = []

    for i in xrange(0, no_out):
        diff = (tvec[i] - outvec2[i]) * diff_sigmoid(outnet[i], 1.0)
        deltak.append(diff)
        for j in xrange(0, no_hid):
            out.set_delta_weight(i, j, eta * diff * hidout[j])

    outweight = out.get_weight()
    length_out = len(outweight)
    for j in xrange(0, no_hid):
        sumterm = 0
        for k in xrange(0, length_out):
            sumterm += (deltak[k] * outweight[k][j])
        deltaj = sumterm * diff_sigmoid(hidnet[j], 1.0)
        for i in xrange(0, len(inp) - 1):
            hid.set_delta_weight(j, i, eta * deltaj * inp[i])

def rms(tvec, inp, hid, out, no_hid, no_out):
    result = 0
    count = 0
    for i in xrange(0, len(inp)):
        feed_forward(inp[i], hid, out, no_hid, no_out)
        outvec = out.get_set_out()
        out2 = out.get_output()
        result += (pow((tvec[i][0] - outvec[0]), 2) + pow(tvec[i][1] - outvec[1], 2))
        if outvec == tvec[i]:
            count += 1
        print out2
    result = result / len(inp)
    return math.sqrt(result), (float(count) / float(len(inp))) * 100

def run_iterations():
    inputlayer = input_layer('test.txt')
    num_hid = 4
    hiddenlayer = hidden_layer(num_hid, 65)
    outputlayer = output_layer(2, num_hid)
   
    inputvectors = inputlayer.get_vectors()
    originalvectors = inputlayer.get_tk()
    count = 0
    while 1:
        count += 1
        hiddenlayer.reset_delta_weight()
        outputlayer.reset_delta_weight()
        for i in xrange(0, len(inputvectors)):
            feed_forward(inputvectors[i], hiddenlayer, outputlayer, num_hid, 2)
            back_propagation(inputvectors[i], hiddenlayer, outputlayer, num_hid, 2, originalvectors[i])
        hiddenlayer.set_weight()
        outputlayer.set_weight()
        if count % 100 == 0:
            count = 0
            result, cmon = rms(originalvectors, inputvectors, hiddenlayer, outputlayer, num_hid, 2)
            if result <= 0.01:
                break
            print cmon
            print "hidden", hiddenlayer.get_weight()
            print "output", outputlayer.get_weight()
    print "HIDDEN WEIGHTS"
    print hiddenlayer.get_weight()
    print "OUTPUT WEIGHTS"
    print outputlayer.get_weight()

run_iterations()
        
        
