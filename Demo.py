4# demostrate how to invoke the WFO.py file

from WFO import WFO
from matplotlib import pyplot
from CEC2022 import cec22_test_func

from math import pi, cos


# set the algorithmic parameters
class alg:
    NP = 20
    max_nfe = 10000
    pl = 0.7
    pe = 0.3
    
# set the problem's parameters
n = 10   # dimensional number
class prob:
    dim = n
    lb = [-100]*n
    ub = [100]*n
    
    # the objective function to be minimized,
    # for example 1 , uni-mode sphere function, f = sum(x^2)
#    def fobj(x): 
 #       f = 0
 #       for i in x:
 #           f += i**2
 #       return f
    
    # for example 2, multi-mode Rastrign function, f =sum(x^2-10*cos(2*pi*x)) + 10*n  
    #def fobj(x): 
      #  f = 0
        #for i in x:
          #  f += i**2 - 10*cos(2*pi*i)
        #f += 10*n
        #return f


     #for example 3, CEC2022 single objective benchmark problems
    def fobj(x):        
        f = cec22_test_func(x = x, nx = n, mx = 1, func_num = 2)
        return f
    
fb, xb, con = WFO(alg, prob)
print('The minimal objective function value: {}'.format(fb))
print('The best solution:')
print(xb)

pyplot.plot(con)
pyplot.xlabel('Number of function evaluation')
pyplot.ylabel('Function value')
pyplot.title('Convergence')
pyplot.show()
