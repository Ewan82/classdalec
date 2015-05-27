"""Tests for the functions in the fourdvar module.
"""
import numpy as np
import ahdata2 as dC
import modclass as mc
import matplotlib.pyplot as plt


def test_costfn(alph=1e-9):
    """Test for cost and gradcost functions.
    """
    d = dC.DalecData(50, 'nee')
    m = mc.DalecModel(d)
    gradj = m.gradcost(d.pvals)
    h = gradj*(np.linalg.norm(gradj))**(-1)
    j = m.cost(d.pvals)
    jalph = m.cost(d.pvals + alph*h)
    print (jalph-j) / (np.dot(alph*h, gradj))
    assert (jalph-j) / (np.dot(alph*h, gradj)) < 1.0001

def test_cost(alph=1e-8, vect=0):
    """Test for cost and gradcost functions.
    """
    d = dC.DalecData(365, 'nee')
    m = mc.DalecModel(d)
    gradj = m.gradcost(d.pvals)
    if vect == True:
        h = d.pvals*(np.linalg.norm(d.pvals))**(-1)
    else:
        h = gradj*(np.linalg.norm(gradj))**(-1)
    j = m.cost(d.pvals)
    jalph = m.cost(d.pvals + alph*h)
    print jalph
    print j
    print np.dot(alph*h, gradj)
    return (jalph-j) / (np.dot(alph*h, gradj))

def plotcost():
    """Using test_cost plots convergance of cost fn gradient for decreasing
    value of alpha.
    """
    power=np.arange(-2,7,1)
    xlist = [10**(-x) for x in power]
    tstlist = [abs(1-test_cost(x, 1)) for x in xlist]
    plt.loglog(xlist, tstlist)
    plt.xlabel('alpha')
    plt.ylabel('abs(1 - grad test function)')
    plt.title('test of the gradient of the cost function')
    print tstlist
    plt.show()