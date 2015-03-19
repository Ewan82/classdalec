import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import model as m
import observations as obs


def sic(dC, ob, pvals, x):
    """Given a dataClass, an obsevation as a string, a linearized model 
    matrix, a set of parameter values and a timestep (x) calculates the shannon
    information content.
    """
    rmat = dC.errdict[ob]**2
    hmat = obs.linob(ob, pvals, dC, x)    
    bmat = dC.B
    jmat = np.linalg.inv(bmat) + np.dot(np.dot(hmat.T, rmat**-1), hmat)
    sic = 0.5*np.log(np.linalg.det(bmat)*np.linalg.det(jmat))
    return sic
    

def siclist(dC, ob, pvals, start, fin):
    """Given a dataClass, an obsevation as a string, a linearized model 
    matrix, a set of parameter values, a start value and a finish value
    calculates a list of SIC values.
    """
    pvallist = m.mod_list(pvals, dC, start, fin)
    siclist = np.ones(fin-start)*-9999.
    
    for x in xrange(fin-start):
        siclist[x] = sic(dC, ob, pvallist[x], x)
        
    return siclist
    
    
def plotsiclist(dC, ob, pvals, start, fin):
    """Plots siclist.
    """
    xlist = np.arange(start,fin)
    datum = dt.datetime(int(dC.year[0]), 1, 1)
    delta = dt.timedelta(hours=24)
    
    times = []
    for t in xlist:
        times.append(datum + int(t) * delta)
    
    siclit = siclist(dC, ob, pvals, start, fin)
    obsat = dC.obdict[ob][start:fin]/dC.obdict[ob][start:fin]
    
    plt.plot(times, siclit*obsat, 'x', markeredgewidth=1.75, 
             markeredgecolor='c')
             
        
    
    
    