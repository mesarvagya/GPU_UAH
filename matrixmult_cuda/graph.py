from matplotlib import pyplot as plt
x = [16,32,64,128,256,512,1024,2048]
y_cpu = [0.04613, 0.352051,2.7641,29.8229, 259.674, 2196.311, 58581.63, 597160.83]
y_gpu = [44.13,48.29,48.355,49.7758, 103.90, 454.395, 632.06,16533.92]
assert(len(y_gpu) == len(y_cpu))
l1 = plt.plot(x, y_cpu, "-g",label="CPU Time")
l2 = plt.plot(x, y_gpu, "-r",label="GPU Time")
plt.xlabel("Matrix Size")
plt.ylabel("Time in ms")
plt.title("GPU vs CPU Matrix Multiplication")
plt.legend(loc="upper center", shadow=True)
plt.show()