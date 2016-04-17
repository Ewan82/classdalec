import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import model as m
import observations as obs
import twindata_ahd as twd
import modclass as mc


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


def ic_obs(ob_str, no_obs, lenwind, pvals):
    """
    Calculates DFS and SIC given:
    :param ob_str: string of observations comma separated
    :param no_obs: list of no. of each ob
    :param lenwind: length of window for obs to be distributed in
    :param pvals: parameter values to linearise about
    :return: SIC, DFS
    """
    d = twd.dalecData(lenwind, ob_str, no_obs)
    d.B = d.makeb(d.test_stdev)
    m = mc.DalecModel(d)
    sic = m.sic(pvals)
    dfs = m.dfs(pvals)
    return sic, dfs
             
        
    
    
    