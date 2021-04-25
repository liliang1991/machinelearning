import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    x = np.arange(-5,5,0.01)

    ## 底数2.718281828459045
    print(np.exp(1))
    y = 1/(1+np.exp(-x))


    plt.plot(x,y)
    plt.xlabel('z')
    plt.ylabel('y')
    plt.grid()
    plt.show()