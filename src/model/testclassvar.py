"""Tests for the functions in the fourdvar module.
"""
import numpy as np
import ahdata2 as dC
import modclass as mc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth':1, 'lines.markersize':6})
    fig, ax = plt.subplots(nrows=1, ncols=1,)#figsize=(10,10))
    sns.set_style('ticks')
    power=np.arange(2,9,1)
    xlist = [10**(-x) for x in power]
    tstlist = [test_linmod(x) for x in xlist]
    ax.loglog(xlist, tstlist, 'k', marker='x', mew=1, ms=8)
    #font = {'size'   : 24}
    #matplotlib.rc('font', **font)
    plt.xlabel(r'$\gamma$')
    plt.ylabel('TLM test function')
    #plt.title('test of the tangent linear model')
    print tstlist
    return ax, fig

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
    pvals = d.edinburghmean
    gradj = m.gradcost2(pvals)
    if vect == True:
        h = pvals*(np.linalg.norm(pvals))**(-1)
    else:
        h = gradj*(np.linalg.norm(gradj))**(-1)
    j = m.cost(pvals)
    jalph = m.cost(pvals + alph*h)
    print jalph - j
    print np.dot(alph*h, gradj)
    return (jalph-j) / (np.dot(alph*h, gradj))

def plotcost(vect=1):
    """Using test_cost plots convergance of cost fn gradient for decreasing
    value of alpha.
    """
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth':1, 'lines.markersize':6})
    fig, ax = plt.subplots(nrows=1, ncols=1,)#figsize=(10,10))
    sns.set_style('ticks')
    power=np.arange(1,10,1)
    xlist = [10**(-x) for x in power]
    tstlist = [abs(test_cost(x, vect)-1) for x in xlist]
    ax.loglog(xlist, tstlist, 'k')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$|1 - f(\alpha)|$')
    #plt.title('test of the gradient of the cost function')
    print tstlist
    return ax, fig

def plotcostone(vect=1):
    """Using test_cost plots convergance of cost fn gradient for decreasing
    value of alpha.
    """
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth':1, 'lines.markersize':6})
    fig, ax = plt.subplots(nrows=1, ncols=1,)#figsize=(10,10))
    sns.set_style('ticks')
    power=np.arange(1,10,1)
    xlist = [10**(-x) for x in power]
    tstlist = [test_cost(x, vect) for x in xlist]
    ax.semilogx(xlist, tstlist, 'k')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$f(\alpha)$')
    plt.ylim((0.9960, 1.0005))
    #plt.title('test of the gradient of the cost function')
    print tstlist
    return ax, fig

def test_cost_cvt(alph=1e-8, vect=0):
    """Test for cost and gradcost functions.
    """
    d = dC.DalecData(365, 'nee')
    m = mc.DalecModel(d)
    pvals = d.edinburghmean
    zvals = m.pvals2zvals(pvals)
    gradj = m.gradcost_cvt(zvals)
    if vect == 0:
        h = zvals*(np.linalg.norm(zvals))**(-1)
    elif vect == 1:
        h = gradj*(np.linalg.norm(gradj))**(-1)
    elif vect == 2:
        h = np.ones(23)*(np.sqrt(23)**-1)
    j = m.cost_cvt(zvals)
    jalph = m.cost_cvt(zvals + alph*h)
    print jalph - j
    print np.dot(alph*h, gradj)
    print (jalph-j) / (np.dot(alph*h, gradj))
    return abs(jalph-j) / (np.dot(alph*h, gradj))

def plotcost_cvt(vect=1):
    """Using test_cost plots convergance of cost fn gradient for decreasing
    value of alpha.
    """
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth':1, 'lines.markersize':6})
    fig, ax = plt.subplots(nrows=1, ncols=1,)#figsize=(10,10))
    sns.set_style('ticks')
    power=np.arange(1,10,1)
    xlist = [10**(-x) for x in power]
    tstlist = [abs(test_cost_cvt(x, vect)-1) for x in xlist]
    ax.loglog(xlist, tstlist, 'k', marker='x', mew=1, ms=8)
    #font = {'size'   : 24}
    #matplotlib.rc('font', **font)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$|f(\alpha) - 1|$')
    #plt.title('test of the gradient of the cost function')
    print tstlist
    #plt.show()
    return ax, fig

def plotcostone_cvt(vect=1):
    """Using test_cost plots convergance of cost fn gradient for decreasing
    value of alpha.
    """
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth': 1, 'lines.markersize': 6})
    fig, ax = plt.subplots(nrows=1, ncols=1,)#figsize=(10,10))
    sns.set_style('ticks')
    power=np.arange(1,14,1)
    xlist = [10**(-x) for x in power]
    tstlist = [test_cost_cvt(x, vect) for x in xlist]
    ax.semilogx(xlist, tstlist, 'k', marker='x', mew=1, ms=8)
    #ax.semilogx(xlist, tstlist, 'k', 'x')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$f(\alpha)$')
    #plt.title('test of the gradient of the cost function')
    print tstlist
    return ax, fig