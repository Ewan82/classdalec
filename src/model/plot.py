"""Plotting functions related to dalecv2.
"""
import numpy as np
import matplotlib.pyplot as plt
import model as m
import observations as obs
import copy as cp
 
    
def plotgpp(cf, dC, start, fin):
    """Plots gpp using acm equations given a cf value, a dataClass and a start
    and finish point. NOTE cf is treated as constant in this plot 
    (unrealistic).
    """
    xlist = np.arange(start, fin, 1)
    gpp = np.ones(fin - start)*-9999.
    for x in xrange(start, fin):
        gpp[x-start] = m.acm(cf, dC.p17, dC.p11, dC, x)
    plt.plot(xlist, gpp)
    plt.show()
    
    
def plotphi(onoff, dC, start, fin):
    """Plots phi using phi equations given a string "fall" or "onset", a 
    dataClass and a start and finish point. Nice check to see dynamics.
    """
    xlist = np.arange(start, fin, 1)
    phi = np.ones(fin - start)*-9999.
    for x in xrange(start, fin):
        if onoff == 'onset':
            phi[x-start] = m.phi_onset(dC.p12, dC.p14, dC, x)
        elif onoff == 'fall':
            phi[x-start] = m.phi_fall(dC.p15, dC.p16, dC.p5, dC, x)
    plt.plot(xlist, phi)
    plt.show()    
    
    
def plotobs(ob, pvals, dC, start, fin, lab=0, dashed=0):
    """Plots a specified observation using obs eqn in obs module. Takes an
    observation string, a dataClass (dC) and a start and finish point.
    """
    modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 'cf': obs.cf,
                 'clab': obs.clab, 'cr': obs.cr, 'cw': obs.cw, 'cl': obs.cl,
                 'cs': obs.cs, 'lf': obs.lf, 'lw': obs.lw, 'lai':obs.lai,
                 'soilresp': obs.soilresp, 'litresp': obs.litresp}
    if lab == 0:
        lab = ob
    else:
        lab = lab
    pvallist = m.mod_list(pvals, dC, start, fin)
    xlist = np.arange(start, fin)
    oblist = np.ones(fin - start)*-9999.
    for x in xrange(start, fin):
        oblist[x-start] = modobdict[ob](pvallist[x-start],dC,x)
    if dashed==True:    
        plt.plot(xlist, oblist, '--', label=lab)
    else:
        plt.plot(xlist, oblist, label=lab)
        

def plot4dvarrun(ob, xb, xa, dC, start, fin, erbars=1):
    """Plots a model predicted observation value for two initial states (xb,xa)
    and also the actual observations taken of the physical quantity. Takes a ob
    string, two initial states (xb,xa), a dataClass and a start and finish 
    time step.
    """
    xlist = np.arange(start, fin)
    plotobs(ob, xb, dC, start, fin, ob+'_b')
    plotobs(ob, xa, dC, start, fin, ob+'_a', 1)
    obdict = dC.obdict
    oberrdict = dC.oberrdict
    if ob in obdict.keys():
        if erbars==True:
            plt.errorbar(xlist, obdict[ob], yerr=oberrdict[ob+'_err'], \
                         fmt='o', label=ob+'_o')
        else:
            plt.plot(xlist, obdict[ob], 'o', label=ob+'_o')
    plt.legend()
    plt.show()
    
    
def plottwin(ob, truth, xb, xa, dC, start, fin, erbars=1, obdict=None,
             oberrdict=None):
    """For identical twin experiments plots the truths trajectory and xa's and 
    xb's to see imporvements.
    """
    xlist = np.arange(start, fin)
    plotobs(ob, truth, dC, start, fin, ob+'_truth')
    plotobs(ob, xb, dC, start, fin, ob+'_b')
    plotobs(ob, xa, dC, start, fin, ob+'_a', 1)
    
    if obdict == None:
        obdict = dC.obdict
        oberrdict = dC.oberrdict
    else:
        obdict = obdict
        oberrdict = oberrdict
        
    if ob in obdict.keys() and len(obdict[ob])==(start-fin) and erbars==True:
        plt.errorbar(xlist, obdict[ob], yerr=oberrdict[ob+'_err'], \
                         fmt='o', label=ob+'_o')
    elif ob in obdict.keys() and len(obdict[ob])!=(start-fin) and erbars==True:
        apnd = np.ones((fin-start)-len(obdict[ob]))*float('NaN')
        obdict = np.concatenate((obdict[ob],apnd))                
        oberrdict = np.concatenate((oberrdict[ob+'_err'],apnd)) 
        plt.errorbar(xlist, obdict, yerr=oberrdict, \
                         fmt='o', label=ob+'_o')
                         
    elif ob in obdict.keys() and len(obdict[ob])==(start-fin) and erbars==False:       
        plt.plot(xlist, obdict[ob], 'o', label=ob+'_o')
    elif ob in obdict.keys() and len(obdict[ob])!=(start-fin) and erbars==False:
        apnd = np.ones((fin-start)-len(obdict[ob]))*float('NaN')
        obdict = np.concatenate((obdict[ob],apnd))                
        plt.plot(xlist, obdict, 'o', label=ob+'_o')
        
    plt.xlabel('Day')
    plt.ylabel(ob)
    plt.title(ob+' plotted for truth, xb and xa.')
    plt.legend()
    plt.show() 
    
    
def plottwinerr(truth, xb, xa):
    """Plot error between truth and xa/xb shows as a bar chart.
    """
    n = 23
    width = 0.35
    ind = np.arange(n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, 100*abs(truth-xb)/xb, width, color='r',\
                    label='xb_err')
    rects2 = ax.bar(ind+width, 100*abs(truth-xa)/xa, width, color='g',\
                    label='xa_err')
    ax.set_ylabel('% error')
    ax.set_title('% error in parameter values for xa and xb')
    ax.set_xticks(ind+width)
    keys = ['theta_min', 'f_auto', 'f_fol', 'f_roo', 'clspan', 'theta_woo',
            'theta_roo', 'theta_lit', 'theta_som', 'Theta', 'ceff', 'd_onset',
            'f_lab', 'cronset', 'd_fall', 'crfall', 'clma', 'clab', 'cf', 'cr',
            'cw', 'cl', 'cs']
    ax.set_xticklabels(keys, rotation=90)
    ax.legend()
    plt.show()

    
def plotlinmoderr(cpool, dC, start, fin):
    """Plots the error for the linearized estimate to the evolution of a carbon
    pool and the nonlinear models evolution of a carbon pool for comparision 
    and to see if the linear model satisfies the tangent linear hypoethesis.
    Takes a carbon pool as a string, a dataClass and a start and finish point.
    """
    pooldict={'clab':0, 'cf':1, 'cr':2, 'cw':3, 'cl':4, 'cs':5}
    cx, matlist = m.linmod_list(dC.pvals, dC, start, fin)
    dC2 = cp.copy(dC)
    dC2.pvals = dC2.pvals*1.05
    cxdx = m.mod_list(dC2.pvals, dC2, start, fin)
    dC3 = cp.copy(dC)
    dC3.pvals = dC3.pvals*0.05
    
    dxl = m.linmod_evolvefac(dC3.pvals, matlist, dC, start, fin)
    
    dxn = cxdx-cx
    
    plt.plot(dxn[:,pooldict[cpool]],label='dxn '+cpool)
    plt.plot(dxl[:,pooldict[cpool]],label='dxl '+cpool)
    plt.legend()
    plt.show()
    
