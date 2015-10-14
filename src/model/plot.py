"""Plotting functions related to dalecv2.
"""
import numpy as np
import matplotlib.pyplot as plt
import model as m
import fourdvar as fdv
import observations as obs
import copy as cp
import datetime as dt
import taylordiagram as td
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
import matplotlib
import seaborn as sns


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
            colour=None, ax=None):
    """Plots a specified observation using obs eqn in obs module. Takes an
    observation string, a dataClass (dC) and a start and finish point.
    """
    sns.set_context(rc={'lines.linewidth':0.4, 'lines.markersize':3.8})
    if ax == None:
        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,10))
    else:
        ax = ax
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
            ax.plot(xax, oblist, '--', label=lab)
        else:
            ax.plot(xax, oblist, label=lab)
    else:
        if dashed==True:    
            ax.plot(xax, oblist, '--', label=lab, color=colour)
        else:
            ax.plot(xax, oblist, label=lab, color=colour)
    return ax


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
    sns.set_style('ticks')
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth':0.4, 'lines.markersize':3.8})
    fig, ax = plt.subplots(nrows=1, ncols=1,) #figsize=(20,10))
    xlist = np.arange(start, fin)
    # We know the datum and delta from reading the file manually
    datum = dt.datetime(int(dC.year[0]), 1, 1)
    delta = dt.timedelta(hours=24)
    
    # Convert the time values to datetime objects
    times = []
    for t in xlist:
        times.append(datum + int(t) * delta)

    ax2 = plotobs(ob, xb, dC, start, fin, ob+'_b', times, 1, ax=ax)
    ax3 = plotobs(ob, xa, dC, start, fin, ob+'_a', times, ax=ax2)
    obdict = dC.obdict
    oberrdict = dC.oberrdict
    if ob in obdict.keys():
        if erbars==True:
            ax3.errorbar(times, obdict[ob], yerr=oberrdict[ob+'_err'], \
                         fmt='o', label=ob+'_o')
        else:
            ax3.plot(times, obdict[ob], 'o', label=ob+'_o')
    if obdict_a!=None:
        ax3.plt.plot(times[0:len(obdict_a[ob])], obdict_a[ob], 'o')
    #ax3.legend()
    plt.xlabel('Year')
    plt.ylabel(ob.upper()+' (g C m-2)')
    #plt.title(ob+' for Alice Holt flux site')
    plt.ylim((-20,15))
    if awindl!=None:
        ax3.axvline(x=times[awindl],color='k',ls='dashed')
        #ax3.text(times[20], 8.5, 'Assimilation\n window')
        #ax3.text(times[awindl+20], 9, 'Forecast')

    #plt.gcf().autofmt_xdate()     
    #ax = plt.gca()
    #ax.autofmt_xdate()
    #ax = plt.gca()
    #ax.xaxis.set_major_locator(dayLocator)
    #ax.xaxis.set_major_formatter(dateFmt)
    #ax.xaxis.set_minor_locator(hourLocator)
    return ax3, fig
    """plt.show()
    import matplotlib.pyplot as plt
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.plot([0,1,2], [10,20,3])
fig.savefig('path/to/save/image/to.png')   # save the figure to file
plt.close(fig)    # close the figure"""
    
    
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
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth':1, 'lines.markersize':6})
    fig, ax = plt.subplots(nrows=1, ncols=1,)#figsize=(10,10))
    sns.set_style('ticks')
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
        ax.plot(y[0:splitval], hx[0:splitval], 'o')
        error = np.sqrt(np.sum((y[0:splitval]-hx[0:splitval])**2)/\
                                                            len(y[0:splitval]))
        yhx = np.mean(y[0:splitval]-hx[0:splitval])
    elif bfa=='f':
        ax.plot(y[splitval:], hx[splitval:], 'o')
        error = np.sqrt(np.sum((y[splitval:]-hx[splitval:])**2)/\
                                                            len(y[splitval:]))
        yhx = np.mean(y[splitval:]-hx[splitval:])                                            
    else:
        raise Exception('Please check function input for bfa variable')
    plt.xlabel(ob.upper()+' observations (g C m-2)')
    plt.ylabel(ob.upper()+' model (g C m-2)')
    #plt.title(bfa+'_error=%f, mean(y-hx)=%f' %(error,yhx))
    print bfa+'_error=%f, mean(y-hx)=%f' %(error,yhx)
    plt.xlim((-20,10))
    plt.ylim((-20,15))
    return ax, fig #error, y[0:splitval]-hx[0:splitval]


def da_std_corrcoef_obs(ob, pvals, dC, awindl, bfa='a'):
    """
    :param ob: observation to compare with data
    :param pvals: xa from DA experiment
    :param dC: dataClass with observations
    :param awindl: length of analysis window
    :param bfa: background, forecast or analysis (b,f,a)
    :return: std of model and observations, rms and correlation coef.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1,)
    pvallist = m.mod_list(pvals, dC, 0, len(dC.I))
    y, yerr = fdv.obscost(dC.obdict, dC.oberrdict)
    hx = fdv.hxcost(pvallist, dC.obdict, dC)
    splitval = 0
    for x in xrange(0,awindl):
        if np.isnan(dC.obdict[ob][x])!=True:
            splitval += 1

    if bfa=='b' or bfa=='a':
        obs = y[0:splitval]
        mod_obs = hx[0:splitval]
    elif bfa=='f':
        obs = y[splitval:]
        mod_obs = hx[splitval:]
    else:
        raise Exception('Please check function input for bfa variable')

    std_mod_obs = np.std(mod_obs)
    mod_obs_bar = np.mean(mod_obs)
    std_obs = np.std(obs)
    obs_bar = np.mean(obs)
    rms = np.sqrt(np.sum([((mod_obs[x]-mod_obs_bar)-(obs[x]-obs_bar))**2 for x in xrange(len(obs))])/len(obs))
    corr_coef = (np.sum([((mod_obs[x]-mod_obs_bar)*(obs[x]-obs_bar)) for x in xrange(len(obs))])/len(obs))\
                /(std_mod_obs*std_obs)
    return std_mod_obs, std_obs, rms, corr_coef

def plot_taylor_diagram():
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth':1.4, 'lines.markersize':9})
    fig = plt.figure()
    sns.set_style('ticks')
    experiments = [(2.3484143513774565, 0.66197831433922605, 'xb'),
                   (6.7512375011313743, 0.78718349020688749, 'A'),
                   (5.1295310098679368, 0.8684810131720303, 'B'),
                   (6.49605967589775, 0.78273625256663071, 'C'),
                   (4.9966680670866221, 0.88331243820904048, 'D')]
    std_obs = 4.7067407809222761
    dia = td.TaylorDiagram(std_obs, fig=fig)
    for i, (std, corrcoef, name) in enumerate(experiments):
        dia.add_sample(std, corrcoef, marker='o', label=name)
    contours = dia.add_contours(levels=8)
    plt.clabel(contours, inline=1, fontsize=10)
    fig.legend(dia.samplePoints,
               [p.get_label() for p in dia.samplePoints],
               numpoints=1, loc='upper right')
    plt.ylabel('Standard Deviation')
    plt.show()

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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    xlist = np.arange(start, fin)
    ax = plotobs(ob, truth, dC, start, fin, ob+'_truth', ax=ax)
    ax = plotobs(ob, xb, dC, start, fin, ob+'_b', ax=ax)
    ax = plotobs(ob, xa, dC, start, fin, ob+'_a', None, 1, ax=ax)
    
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
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111)
    #ax.bar(ind, ((xa-xb)/abs(xa-xb))*np.log(abs(xa-xb)), width, color='g',\
    #                label='xa_inc')
    ax.bar(ind, (xa-xb), width, color='g',\
                   label='xa_inc')
    ax.set_ylabel('xa - xb')
    ax.set_title('Analysis increment')
    ax.set_xticks(ind)
    keys = ['theta_min', 'f_auto', 'f_fol', 'f_roo', 'clspan', 'theta_woo',
            'theta_roo', 'theta_lit', 'theta_som', 'Theta', 'ceff', 'd_onset',
            'f_lab', 'cronset', 'd_fall', 'crfall', 'clma', 'clab', 'cf', 'cr',
            'cw', 'cl', 'cs']
    ax.set_xticklabels(keys, rotation=90)
    return ax, fig
    
def plotbmat(bmat):
    """Plots a B matrix.
    """
    sns.set(style="whitegrid")
    sns.set_context('poster', font_scale=1.2)
    fig, ax = plt.subplots(figsize=(11,9))
    ax.set_aspect('equal')
    keys = ['theta_min', 'f_auto', 'f_fol', 'f_roo', 'clspan', 'theta_woo',
            'theta_roo', 'theta_lit', 'theta_som', 'Theta', 'ceff', 'd_onset',
            'f_lab', 'cronset', 'd_fall', 'crfall', 'clma', 'clab', 'cf', 'cr',
            'cw', 'cl', 'cs']
    ax.set_xticks(np.arange(23))
    ax.set_xticklabels(keys, rotation=90)
    ax.set_yticks(np.arange(23))
    ax.set_yticklabels(keys)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.eye(23, dtype=bool)
    sns.heatmap(bmat, mask=mask, xticklabels=keys, yticklabels=keys, ax=ax,
                cmap=cmap, vmax=.43, square=True, linewidths=.5, cbar=True)

    #plt.colorbar()
    return ax, fig

# Draw the heatmap with the mask and correct aspect ratio
#sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
#            square=True, xticklabels=5, yticklabels=5,
#            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

def plotrmat(rmat):
    """Plots a R matrix.
    """
    sns.set(style="white")
    sns.set_context('poster', font_scale=1.2)
    fig, ax = plt.subplots(figsize=(11,9))
    ax.set_aspect('equal')
    sns.heatmap(rmat, ax=ax, vmax=.5, xticklabels=False, yticklabels=False,
                square=True, linewidths=.5, cbar=True)
    #sns.heatmap(rmat, ax=ax, xticklabels=np.arange(len(rmat)), yticklabels=np.arange(len(rmat)))
    return ax, fig
    
def analysischange(xb, xa):
    """Plot error between truth and xa/xb shows as a bar chart.
    """
    n = 23
    width = 0.35
    ind = np.arange(n)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.barh(ind, (xb-xa)/xb, width, color='r',\
                    label='xa_change')
    ax.set_xlabel('(xb-xa)/xb')
    ax.set_title('change in parameter values from xb to xa')
    ax.set_yticks(ind+width)
    keys = ['theta_min', 'f_auto', 'f_fol', 'f_roo', 'clspan', 'theta_woo',
            'theta_roo', 'theta_lit', 'theta_som', 'Theta', 'ceff', 'd_onset',
            'f_lab', 'cronset', 'd_fall', 'crfall', 'clma', 'clab', 'cf', 'cr',
            'cw', 'cl', 'cs']
    ax.set_yticklabels(keys)
    return ax, fig
    
    
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



