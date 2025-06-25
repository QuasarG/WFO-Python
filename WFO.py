# WFO: Water Flow Optimizer
# Reference: 
# Kaiping Luo. Water Flow Optimizer: a nature-inspired evolutionary algorithm for global optimization.
# IEEE Transactions on Cybernetics, 2021.
# ------------------------------------------------------
# Syntax   WFO.WFO(alg, prob)
#
# alg is a class, which includes the following fields:
# alg.NP:  the number of water particle
# alg.max_nfe: the maximal number of function evaluation
# alg.pl: the laminar probability
# al.pe: the eddying probability
#
# prob is also a class, which includes the following fields:
# prob.lb:  a row vector representing the lower bound
# prob.ub:  a row vector representing the upper bound
# prob.fobj:  a function representing the objective to be minimized
# prob.dim:  the dimension of the given problem
#
# The WFO function will return three arguments:
# fb: the best objective function value found
# xb: the best solution found
# con: convergence
# --------------------------------------------------------
# Edited by: Kaiping Luo, Beihang University, kaipingluo@buaa.edu.cn
# in Python 3

from numpy import zeros,transpose
from math import inf,pi,cos
from random import random,uniform,randrange

def WFO(alg, prob):
    global fb, xb, con
    fb = inf
    con = zeros((alg.max_nfe,1))
    X = zeros((alg.NP, prob.dim))
    F = zeros((alg.NP, 1))
    for i in range(alg.NP): # initialization
        for j in range(prob.dim):
            X[i,j] = uniform(prob.lb[j],prob.ub[j])            
        F[i] = prob.fobj(X[i,:])
        if F[i]<fb:
            fb = F[i]
            xb = X[i,:]            
        con[i] = fb
    Y = zeros((alg.NP, prob.dim))
    nfe = i
    while nfe < alg.max_nfe-1:
        if random() < alg.pl: # laminar flow
            k = randrange(alg.NP)
            d = xb - X[k,:]
            for i in range(alg.NP):
                Y[i,:] = X[i,:] + random()*d
                for j in range(prob.dim):
                    if not prob.lb[j] <= Y[i,j] <= prob.ub[j]:
                        Y[i,j] = X[i,j]
                        
        else: # turbulent flow
            for i in range(alg.NP):
                Y[i,:] = X[i,:]
                k = randrange(alg.NP)
                while k==i:
                    k = randrange(alg.NP)
                j1 = randrange(prob.dim)
                if random() < alg.pe: # spiral flow
                    theta = uniform(-pi, pi)
                    Y[i,j1] = X[i,j1]+abs(X[k,j1]-X[i,j1])*theta*cos(theta)
                    if not prob.lb[j1] <= Y[i,j1] <= prob.ub[j1]:
                        Y[i,j1] = X[i,j1] 
                else:
                    j2 = randrange(prob.dim)
                    while j2==j1:
                        j2 = randrange(prob.dim)
                    Y[i,j1] = prob.lb[j1] + (prob.ub[j1]-prob.lb[j1])*(X[k,j2]-prob.lb[j2])/(prob.ub[j2]-prob.lb[j2])
                    
        for i in range(alg.NP): # evaluation and evolution
            f = prob.fobj(Y[i,:])
            if f < F[i]:
                F[i] = f
                X[i,:] = Y[i,:]
                if f < fb:
                    fb = f
                    xb = X[i,:]
            nfe += 1
            con[nfe] = fb
    return fb, xb,con
