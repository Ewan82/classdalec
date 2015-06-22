"""Tests for the functions in the fourdvar module.
"""
import numpy as np
import ahdata2 as dC
import modclass as mc
import matplotlib
import matplotlib.pyplot as plt

def test_linmod(gamma=1e1):
    """ Test from TLM to check it converges.
    """
    d = dC.DalecData(731, 'nee')
    pvals = d.pvals
    m = mc.DalecModel(d)
    cx, matlist = m.linmod_list(pvals)
    pvals2 = pvals*(1 + 0.3*gamma)
    cxdx = m.mod_list(pvals2)[-1]
    pvals3 = pvals*(0.3*gamma)

    dxl = np.linalg.norm(np.dot(m.mfac(matlist, 730), pvals3.T))

    dxn = np.linalg.norm(cxdx-cx)
    return dxl / dxn

def plt_linmod_er():
    """Plots linmod test for decreasing gamma.
    """
    power=np.arange(1,7,1)
    xlist = [10**(-x) for x in power]
    tstlist = [test_linmod(x) for x in xlist]
    plt.loglog(xlist, tstlist, 'k')
    font = {'size'   : 24}
    matplotlib.rc('font', **font)
    plt.xlabel('Gamma')
    plt.ylabel('TLM test function')
    #plt.title('test of the tangent linear model')
    print tstlist
    plt.show()

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
    pvals = d.edinburghpvals
    gradj = m.gradcost(pvals)
    if vect == True:
        h = pvals*(np.linalg.norm(pvals))**(-1)
    else:
        h = gradj*(np.linalg.norm(gradj))**(-1)
    j = m.cost(pvals)
    jalph = m.cost(pvals + alph*h)
    print jalph - j
    print np.dot(alph*h, gradj)
    return (jalph-j) / (np.dot(alph*h, gradj))

def plotcost():
    """Using test_cost plots convergance of cost fn gradient for decreasing
    value of alpha.
    """
    power=np.arange(0,6,1)
    xlist = [10**(-x) for x in power]
    tstlist = [abs(test_cost(x, 1)-1) for x in xlist]
    plt.loglog(xlist, tstlist, 'k')
    font = {'size'   : 24}
    matplotlib.rc('font', **font)
    plt.xlabel('alpha')
    plt.ylabel('abs(1 - grad test function)')
    #plt.title('test of the gradient of the cost function')
    print tstlist
    plt.show()

def plotcostone():
    """Using test_cost plots convergance of cost fn gradient for decreasing
    value of alpha.
    """
    power=np.arange(0,5,1)
    xlist = [10**(-x) for x in power]
    tstlist = [test_cost(x, 1) for x in xlist]
    plt.semilogx(xlist, tstlist, 'k')
    font = {'size'   : 24}
    matplotlib.rc('font', **font)
    plt.xlabel('alpha')
    plt.ylabel('grad test function')
    #plt.title('test of the gradient of the cost function')
    print tstlist
    plt.show()

def test_cost_cvt(alph=1e-8, vect=0):
    """Test for cost and gradcost functions.
    """
    d = dC.DalecData(365, 'nee')
    m = mc.DalecModel(d)
    pvals = d.edinburghpvals
    zvals = m.pvals2zvals(pvals)
    gradj = m.gradcost_cvt(zvals)
    if vect == True:
        h = zvals*(np.linalg.norm(zvals))**(-1)
    else:
        h = gradj*(np.linalg.norm(gradj))**(-1)
    j = m.cost_cvt(zvals)
    jalph = m.cost_cvt(zvals + alph*h)
    print jalph - j
    print np.dot(alph*h, gradj)
    print (jalph-j) / (np.dot(alph*h, gradj))
    return (jalph-j) / (np.dot(alph*h, gradj))

def plotcost_cvt():
    """Using test_cost plots convergance of cost fn gradient for decreasing
    value of alpha.
    """
    power=np.arange(0,8,1)
    xlist = [10**(-x) for x in power]
    tstlist = [abs(test_cost_cvt(x, 1)-1) for x in xlist]
    plt.loglog(xlist, tstlist, 'k')
    font = {'size'   : 24}
    matplotlib.rc('font', **font)
    plt.xlabel('alpha')
    plt.ylabel('abs(1 - grad test function)')
    #plt.title('test of the gradient of the cost function')
    print tstlist
    plt.show()

def plotcostone_cvt():
    """Using test_cost plots convergance of cost fn gradient for decreasing
    value of alpha.
    """
    power=np.arange(0,8,1)
    xlist = [10**(-x) for x in power]
    tstlist = [test_cost_cvt(x, 1) for x in xlist]
    plt.semilogx(xlist, tstlist, 'k')
    font = {'size'   : 24}
    matplotlib.rc('font', **font)
    plt.xlabel('alpha')
    plt.ylabel('grad test function')
    #plt.title('test of the gradient of the cost function')
    print tstlist
    plt.show()