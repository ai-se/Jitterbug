from __future__ import division, print_function
import numpy as np
from scipy.optimize import minimize


class Sum_Regularized_Regression():
    def __init__(self,beta=np.array([0.0,1.0])):
        self.beta=beta

    def check_input_type(self,x):
        return type(x)==type(np.array([]))

    def logistic(self,x,beta):
        return 1.0/(1.0+np.exp(-(beta[0]+beta[1]*x)))

    def fitness_func(self,x,y,beta):
        p = self.logistic(x,beta)
        return np.linalg.norm(y-p, 2) ** 2+(np.sum(y)-np.sum(p))**2

    def fit(self,x,y):
        if not (self.check_input_type(x) and self.check_input_type(y)):
            print("Input type must be numpy.array.")
            return
        beta0 = self.beta
        # res = minimize(lambda beta:self.fitness_func(x,y,beta), beta0, method='L-BFGS-B', options = {'disp': False})
        res = minimize(lambda beta:self.fitness_func(x,y,beta), beta0, method='nelder-mead', options = {'xatol': 1e-8, 'disp': False})
        self.beta = res.x

    def decision_function(self,x):
        return self.logistic(x,self.beta)

    def predict(self,x):
        return self.decision_function(x)


if __name__ == "__main__":
    from pdb import set_trace
    from sklearn.linear_model import LogisticRegression
    x=np.array([0,1,2,3,4,5,6])
    y=np.array([0,0,0,1,0,1,1])
    model  = Sum_Regularized_Regression()
    model.fit(x,y)
    y0=model.predict(x)
    print(y0)
    print(sum(y0))
    model2 = LogisticRegression()
    x_new = [[xx] for xx in x]
    model2.fit(x_new,y)
    y2 = model2.decision_function(x_new)
    print(y2)
    print(sum(y2))