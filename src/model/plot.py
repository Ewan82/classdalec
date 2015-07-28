"""Plotting functions related to dalecv2.
"""
import numpy as np
import matplotlib.pyplot as plt
import model as m
import fourdvar as fdv
import observations as obs
import copy as cp
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
import matplotlib


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
    
    
def plotobs(ob, pvals, dC, start, fin, lab=0, xax=None, dashed=0, 
            colour=None):
    """Plots a specified observation using obs eqn in obs module. Takes an
    observation string, a dataClass (dC) and a start and finish point.
    """
    modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 'cf': obs.cf,
                 'clab': obs.clab, 'cr': obs.cr, 'cw': obs.cw, 'cl': obs.cl,
                 'cs': obs.cs, 'lf': obs.lf, 'lw': obs.lw, 'lai':obs.lai,
                 'soilresp': obs.soilresp, 'litresp': obs.litresp,
                 'rtot': obs.rtot, 'rh': obs.rh}
    if lab == 0:
        lab = ob
    else:
        lab = lab
    pvallist = m.mod_list(pvals, dC, start, fin)
    if xax==None:
        xax = np.arange(start, fin)
    else:
        xax = xax
    oblist = np.ones(fin - start)*-9999.
    for x in xrange(start, fin):
        oblist[x-start] = modobdict[ob](pvallist[x-start],dC,x)
    if colour==None:
        if dashed==True:    
            plt.plot(xax, oblist, '--', label=lab)
        else:
            plt.plot(xax, oblist, label=lab)
    else:
        if dashed==True:    
            plt.plot(xax, oblist, '--', label=lab, color=colour)
        else:
            plt.plot(xax, oblist, label=lab, color=colour)   
    return oblist


def plot4dvarrun(ob, xb, xa, dC, start, fin, erbars=1, awindl=None, 
                 obdict_a=None):
    """Plots a model predicted observation value for two initial states (xb,xa)
    and also the actual observations taken of the physical quantity. Takes a ob
    string, two initial states (xb,xa), a dataClass and a start and finish 
    time step.
    """
    #dayLocator    = mdates.DayLocator()
    #hourLocator   = mdates.HourLocator()
    #dateFmt = mdates.DateFormatter('%Y')  
    xlist = np.arange(start, fin)
    # We know the datum and delta from reading the file manually
    datum = dt.datetime(int(dC.year[0]), 1, 1)
    delta = dt.timedelta(hours=24)
    
    # Convert the time values to datetime objects
    times = []
    for t in xlist:
        times.append(datum + int(t) * delta)

    plotobs(ob, xb, dC, start, fin, ob+'_b', times, 1)
    plotobs(ob, xa, dC, start, fin, ob+'_a', times)
    obdict = dC.obdict
    oberrdict = dC.oberrdict
    if ob in obdict.keys():
        if erbars==True:
            plt.errorbar(times, obdict[ob], yerr=oberrdict[ob+'_err'], \
                         fmt='o', label=ob+'_o')
        else:
            plt.plot(times, obdict[ob], 'o', label=ob+'_o')
    if obdict_a!=None:
        plt.plot(times[0:len(obdict_a[ob])], obdict_a[ob], 'o')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel(ob+' (gCm-2)')
    plt.title(ob+' for Alice Holt flux site')
    if awindl!=None:
        plt.axvline(x=times[awindl],color='k',ls='dashed')
        plt.text(times[20], 9, 'Assimilation window')
        plt.text(times[awindl+20], 9, 'Forecast')

    #plt.gcf().autofmt_xdate()     
    #ax = plt.gca()
    #ax.autofmt_xdate()
    #ax = plt.gca()
    #ax.xaxis.set_major_locator(dayLocator)
    #ax.xaxis.set_major_formatter(dateFmt)
    #ax.xaxis.set_minor_locator(hourLocator)

    plt.show()
    
    
def plotspasomp(ob, dC, start, fin, xa=None, awindl=None, erbars=1, spa=1):
    """Plot comparison between erics SPA output and the DALEC model run with
    parameters found from 4DVAR run.
    """
    xlist=np.arange(start,fin)
    datum = dt.datetime(int(dC.year[0]), 1, 1)
    delta = dt.timedelta(hours=24)
    
    times = []
    for t in xlist:
        times.append(datum + int(t) * delta)
    
    if xa!=None:    
        plotobs(ob, xa, dC, start, fin, ob+' ACM-D2', times, dashed=0)
    
    if spa==True:
        ericdat=mlab.csv2rec('../../aliceholtdata/ericspadat.csv')
        plt.plot(times, ericdat[ob], '-', label='Eric SPA-D1 '+ob, alpha=0.75)
    
    obdict = dC.obdict
    oberrdict = dC.oberrdict
    
    if ob in obdict.keys():
        if erbars==True:
            plt.errorbar(times, obdict[ob], yerr=oberrdict[ob+'_err'], \
                         fmt='o', label=ob+'_o', color='r', alpha=0.5)
        else:
            plt.plot(times, obdict[ob], 'o', label=ob+'_o', color='r')
    
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel(ob+' (gCm-2)')
    plt.title(ob+' for Alice Holt flux site')
    if awindl!=None:
        plt.axvline(x=times[awindl],color='k',ls='dashed')
        plt.text(times[20], 9, 'Assimilation window')
        plt.text(times[awindl+20], 9, 'Forecast')
    
    plt.show()
    
    
    
def plotscatterobs(ob, pvals, dC, awindl, bfa='a'):
    """Plots scatter plot of obs vs model predicted values. Takes an initial
    parameter set, a dataClass (must have only desired ob for comparison
    specified in dC), assimilation window length and whether a comparison of 
    background 'b', forecast 'f' or analysis 'a' is desired.
    """
    pvallist = m.mod_list(pvals, dC, 0, len(dC.I))
    y, yerr = fdv.obscost(dC.obdict, dC.oberrdict) 
    hx = fdv.hxcost(pvallist, dC.obdict, dC)
    splitval = 0
    for x in xrange(0,awindl):
        if np.isnan(dC.obdict[ob][x])!=True:
            splitval += 1
    oneone=np.arange(int(min(min(y),min(hx)))-1, int(max(max(y),max(hx)))+1)
    plt.plot(oneone, oneone)
    if bfa=='b' or bfa=='a':
        plt.plot(y[0:splitval], hx[0:splitval], 'o')
        error = np.sqrt(np.sum((y[0:splitval]-hx[0:splitval])**2)/\
                                                            len(y[0:splitval]))
        yhx = np.mean(y[0:splitval]-hx[0:splitval])
    elif bfa=='f':
        plt.plot(y[splitval:], hx[splitval:], 'o')
        error = np.sqrt(np.sum((y[splitval:]-hx[splitval:])**2)/\
                                                            len(y[splitval:]))
        yhx = np.mean(y[splitval:]-hx[splitval:])                                            
    else:
        raise Exception('Please check function input for bfa variable')
    plt.xlabel(ob+' observations')
    plt.ylabel(ob+' model')
    plt.title(bfa+'_error=%f, mean(y-hx)=%f' %(error,yhx))
    return error, y[0:splitval]-hx[0:splitval]

    
def plotcycled4dvar(ob, conditions, xb, xa, dC, erbars=1, truth=None):
    """Plots a cycled 4DVar run given an observation string (obs), a set of
    conditions from modclass output and a list of xb's and xa's.
    """
    fin = conditions['lenrun']
    xlist = np.arange(fin)

    if truth!=None:
        plotobs(ob, truth, dC, 0, fin, ob+'truth')

    plotobs(ob, xb[0], dC, conditions['lenwind']*0, fin, ob+'_b', 1)
    for t in xrange(conditions['numbwind']):
        plotobs(ob, xa[t][0], dC, conditions['lenwind']*t, fin, ob+'_a%x' %t)

        plt.axvline(x=t*conditions['lenwind'],color='k',ls='dashed')
        
    obdict = dC.obdict
    oberrdict = dC.oberrdict
    if ob in obdict.keys():
        if erbars==True:
            plt.errorbar(xlist, obdict[ob][0:fin],\
                         yerr=oberrdict[ob+'_err'][0:fin], fmt='o',\
                         label=ob+'_o')
        else:
            plt.plot(xlist, obdict[ob][0:fin], 'o', label=ob+'_o')
            
    plt.xlabel('Day')
    plt.ylabel(ob)
    plt.title(ob+' for cycled 4DVar run')
    plt.legend()
    plt.show()
    
 
def plotsumobs(ob, pvals, dC, start, fin):
    """Plots a specified observation using obs eqn in obs module. Takes an
    observation string, a dataClass (dC) and a start and finish point.
    """
    modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 'cf': obs.cf,
                 'clab': obs.clab, 'cr': obs.cr, 'cw': obs.cw, 'cl': obs.cl,
                 'cs': obs.cs, 'lf': obs.lf, 'lw': obs.lw, 'lai':obs.lai,
                 'soilresp': obs.soilresp, 'litresp': obs.litresp,
                 'rtot': obs.rtot}

    pvallist = m.mod_list(pvals, dC, start, fin)
    xlist = np.arange(start, fin)
    oblist = np.ones(fin - start)*-9999.
    for x in xrange(start, fin):
        oblist[x-start] = modobdict[ob](pvallist[x-start],dC,x)
    return oblist

   
def plotsumcycled4dvar(ob, conditions, xb, xa, dC, truth=None):
    """Plots cumulative sum of modelled observations from pvals taken from
    cycles 4dvar exps.    
    """    
    fin = conditions['lenrun']
    xlist = np.arange(fin)

    if truth!=None:
        sumobs=np.cumsum(plotsumobs(ob, truth, dC, 0, fin))
        plt.plot(np.arange(0, fin),sumobs, label=ob+'_true')
    else:
        sumobs=np.cumsum(plotsumobs(ob, xa[0][0], dC, 0, fin))

    plt.plot(np.arange(0,fin),\
             np.cumsum(plotsumobs(ob, xb[0], dC, conditions['lenwind']*0,\
             fin,)), label=ob+'_b')

    obs=plotsumobs(ob, xa[0][0], dC, 0, fin)
    plt.plot(np.arange(0,fin), np.cumsum(obs), label=ob+'_a0')                
    for t in xrange(1, conditions['numbwind']):
        obs=plotsumobs(ob, xa[t][0], dC, conditions['lenwind']*t, fin)
        obs[0]=obs[0]+sumobs[conditions['lenwind']*t-1]
        plt.plot(np.arange(conditions['lenwind']*t,fin), np.cumsum(obs),\
                 label=ob+'_a%x' %t)
        plt.axvline(x=t*conditions['lenwind'],color='k',ls='dashed')
            
    plt.xlabel('Day')
    plt.ylabel(ob+' cumulative sum')
    plt.title(ob+' for cycled 4DVar run')
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
    plotobs(ob, xa, dC, start, fin, ob+'_a', None, 1)
    
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
    
    
def plotensemtwin(ob, truth, xb, xa, dC, start, fin, erbars=1, obdict=None,
             oberrdict=None):
    """For identical twin experiments plots the truths trajectory and xa's and 
    xb's to see imporvements.
    """
    xlist = np.arange(start, fin)
    plotobs(ob, truth, dC, start, fin, ob+'_truth', 0, 'red')
    for pval in xb:
        plotobs(ob, pval, dC, start, fin, None, 1, 'orange')
    plotobs(ob, np.mean(xb, axis=0), dC, start, fin, ob+'_b', 0, 'blue')
    for pval in xa:
        plotobs(ob, pval, dC, start, fin, None, 1, 'pink')
    plotobs(ob, np.mean(xa, axis=0), dC, start, fin, ob+'_a', 0, 'green')
    
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

def plot_analysis_inc(xb, xa):
    """Plot error between truth and xa/xb shows as a bar chart.
    """
    n = 23
    width = 0.35
    ind = np.arange(n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.bar(ind, ((xa-xb)/abs(xa-xb))*np.log(abs(xa-xb)), width, color='g',\
    #                label='xa_inc')
    ax.bar(ind, (xa-xb)/xb, width, color='g',\
                   label='xa_inc')
    ax.set_ylabel('xa - xb')
    ax.set_title('Analysis increment')
    ax.set_xticks(ind)
    keys = ['theta_min', 'f_auto', 'f_fol', 'f_roo', 'clspan', 'theta_woo',
            'theta_roo', 'theta_lit', 'theta_som', 'Theta', 'ceff', 'd_onset',
            'f_lab', 'cronset', 'd_fall', 'crfall', 'clma', 'clab', 'cf', 'cr',
            'cw', 'cl', 'cs']
    ax.set_xticklabels(keys, rotation=90)
    #ax.legend()
    plt.show()
    
def plotbmat(bmat):
    """Plots a B matrix.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(bmat, interpolation='nearest')   
    plt.colorbar()
    keys = ['theta_min', 'f_auto', 'f_fol', 'f_roo', 'clspan', 'theta_woo',
            'theta_roo', 'theta_lit', 'theta_som', 'Theta', 'ceff', 'd_onset',
            'f_lab', 'cronset', 'd_fall', 'crfall', 'clma', 'clab', 'cf', 'cr',
            'cw', 'cl', 'cs']
    ax.set_xticks(np.arange(23))
    ax.set_xticklabels(keys, rotation=90)
    ax.set_yticks(np.arange(23))
    ax.set_yticklabels(keys)
    plt.show()
    
    
def analysischange(xb, xa):
    """Plot error between truth and xa/xb shows as a bar chart.
    """
    n = 23
    width = 0.35
    ind = np.arange(n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.barh(ind, (xb-xa)/xb, width, color='r',\
                    label='xa_change')
    ax.set_xlabel('diff (xb-xa)/xb')
    ax.set_title('change in parameter values from xb to xa')
    ax.set_yticks(ind+width)
    keys = ['theta_min', 'f_auto', 'f_fol', 'f_roo', 'clspan', 'theta_woo',
            'theta_roo', 'theta_lit', 'theta_som', 'Theta', 'ceff', 'd_onset',
            'f_lab', 'cronset', 'd_fall', 'crfall', 'clma', 'clab', 'cf', 'cr',
            'cw', 'cl', 'cs']
    ax.set_yticklabels(keys)
    #ax.legend()
    plt.show()
    
    
def plotlinmoderr(dC, start, fin, pvals, dx=0.05, cpool=None, norm_err=0, lins='-'):
    """Plots the error for the linearized estimate to the evolution of a carbon
    pool and the nonlinear models evolution of a carbon pool for comparision 
    and to see if the linear model satisfies the tangent linear hypoethesis.
    Takes a carbon pool as a string, a dataClass and a start and finish point.
    """
    pooldict={'clab':-6, 'cf':-5, 'cr':-4, 'cw':-3, 'cl':-2, 'cs':-1}
    cx, matlist = m.linmod_list(pvals, dC, start, fin)
    d2pvals = pvals*(1. + dx)
    cxdx = m.mod_list(d2pvals, dC, start, fin)
    d3pvals = pvals*dx

    dxl = m.linmod_evolvefac(d3pvals, matlist, dC, start, fin)
    
    dxn = cxdx-cx

    xlist = np.arange(start, fin)
    # We know the datum and delta from reading the file manually
    datum = dt.datetime(int(dC.year[0]), 1, 1)
    delta = dt.timedelta(hours=24)

    # Convert the time values to datetime objects
    times = []
    for t in xlist:
        times.append(datum + int(t) * delta)

    # Font change
    font = {'size': 24}

    matplotlib.rc('font', **font)

    if norm_err is True:
        err = np.ones(fin - start)*-9999.
        dxn_norm = np.ones(fin - start)*-9999.
        dxl_norm = np.ones(fin - start)*-9999.
        for x in xrange(fin - start):
            dxn_norm[x] = np.linalg.norm(dxn[x,17:22])
            dxl_norm[x] = np.linalg.norm(dxl[x,17:22])
            err[x] = abs((np.linalg.norm(dxn[x,17:22])/np.linalg.norm(dxl[x,17:22]))-1)*100.
        plt.plot(times, err, 'k', linestyle=lins, label='$\delta x = %.2f $' % dx)
        #plt.plot(times, dxn_norm, label='dxn')
        #plt.plot(times, dxl_norm, label='dxl')
        plt.xlabel('Date')
        plt.ylabel('Percentage error in TLM')
        #plt.title('Plot of the TLM error')
    else:
        plt.plot(dxn[:,pooldict[cpool]],label='dxn '+cpool)
        plt.plot(dxl[:,pooldict[cpool]],label='dxl '+cpool)
    plt.legend(loc=2)



