import matplotlib.pyplot as plt
import numpy as np
import torch

def f(x):
    return np.sin(x)
def get_derivative(x):
    h=1e-4
    return (f(x+h)-f(x))/h
legend=['f(x)=sin(x)',"f'(x)=lim(f(x+h)-f(x))/h"]
axis=plt.gca()
X=np.arange(0,3,0.1)
Y=f(X)
axis.legend=legend
axis.plot(X,Y)
axis.plot(X,get_derivative(X))