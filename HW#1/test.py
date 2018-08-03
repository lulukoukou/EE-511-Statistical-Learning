import numpy as np
from statistics import mean
import matplotlib.pyplot as plt


xs=np.array([1,2,3,4,5])
ys=np.array([1,2,3,4,5])

def best_fit_slope_and_intercept(xs,ys):
    m= (mean(xs)*mean(ys)-mean(xs*ys))/((mean(xs)*mean(xs))-(mean(xs**2)))
    b = mean(ys) - m*mean(xs)
    return m,b

m,b = best_fit_slope_and_intercept(xs, ys)

print (m,b)
one_variable_regression_line=[(m*x)+b for x in xs]
plt.scatter(xs,ys)
plt.plot(xs,one_variable_regression_line)
