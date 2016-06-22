import numpy as np
import matplotlib.pyplot as plt
import modclass_pheno as mcp
import datetime as dt
import seaborn as sns
import matplotlib.dates as mdates


def plotphi(pvals, dC, start, fin):
    """Plots phi using phi equations given a string "fall" or "onset", a
    dataClass and a start and finish point. Nice check to see dynamics.
    """
    m = mcp.DalecModel(dC)
    m.daybb_arr = m.day_bb(pvals[11])
    xlist = np.arange(start, fin, 1)
    phion = np.ones(fin - start)*-9999.
    phioff = np.ones(fin - start)*-9999.
    for x in xrange(start, fin):
        m.x = x
        phion[x-start] = m.phi_onset(m.daybb_arr[x], pvals[13])
        phioff[x-start] = m.phi_fall(pvals[14], pvals[15], pvals[4])
    m.x -= fin
    plt.plot(xlist, phion)
    plt.plot(xlist, phioff)
    plt.show()


def oblist(ob, dC, pvals):
    m = mcp.DalecModel(dC)
    mod_list = m.mod_list(pvals)
    ob_list = m.oblist(ob, mod_list)
    return ob_list


def plotobs(ob, pvals, dC, lab=0, xax=None, dashed=0,
            colour=None, ax=None, nee_std=None):
    """Plots a specified observation using obs eqn in obs module. Takes an
    observation string, a dataClass (dC) and a start and finish point.
    """
    sns.set_context(rc={'lines.linewidth':.8, 'lines.markersize':6})
    palette = sns.color_palette("Greys")
    if ax == None:
        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,10))
    else:
        ax = ax
    m = mcp.DalecModel(dC)
    mod_list = m.mod_list(pvals)
    ob_list = m.oblist(ob, mod_list)
    if lab == 0:
        lab = ob
    else:
        lab = lab
    if xax == None:
        xax = np.arange(len(dC.year))
    else:
        xax = xax
    if colour == None:
        if dashed == True:
            ax.plot(xax, ob_list, '--', label=lab)
        else:
            ax.plot(xax, ob_list, label=lab)
    else:
        if dashed == True:
            ax.plot(xax, ob_list, '--', label=lab, color=colour, alpha=1)
        else:
            ax.plot(xax, ob_list, label=lab, color=colour, alpha=1)
            if nee_std is not None:
                ax.fill_between(xax, ob_list-3*nee_std, ob_list+3*nee_std, facecolor=palette[2],
                                alpha=0.7, linewidth=0.0)
    return ax


def plot_4d_var(ob, dC, xb=None, xa=None, erbars=1, awindl=None,
                 obdict_a=None, nee_std=None):
    """Plots a model predicted observation value for two initial states (xb,xa)
    and also the actual observations taken of the physical quantity. Takes a ob
    string, two initial states (xb,xa), a dataClass and a start and finish
    time step.
    """
    sns.set_style('ticks')
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth':.8, 'lines.markersize':3.8})
    fig, ax = plt.subplots(nrows=1, ncols=1,)
    x_list = np.arange(len(dC.year))
    palette = sns.color_palette("colorblind", 11)
    # We know the datum and delta from reading the file manually
    datum = dt.datetime(int(dC.year[0]), 1, 1)
    delta = dt.timedelta(hours=24)
    # Convert the time values to datetime objects
    times = []
    for t in x_list:
        times.append(datum + int(t) * delta)

    if xb != None:
        ax2 = plotobs(ob, xb, dC, ob+'_b', times, 1, ax=ax,
                      colour=palette[0], nee_std=nee_std)
        ax = ax2
    if xa != None:
        ax3 = plotobs(ob, xa, dC, ob+'_a', times, ax=ax,
                      colour=palette[1], nee_std=nee_std)
        ax = ax3

    obdict = dC.obdict
    oberrdict = dC.oberrdict
    if ob in obdict.keys():
        if erbars==True:
            ax.errorbar(times, obdict[ob], yerr=oberrdict[ob+'_err'],
                         fmt='o', label=ob+'_o', color=palette[2], alpha=0.7)
        else:
            ax.plot(times, obdict[ob], 'o', label=ob+'_o')
    if obdict_a!=None:
        ax.plt.plot(times[0:len(obdict_a[ob])], obdict_a[ob], 'o')

    plt.xlabel('Year')
    plt.ylabel(ob.upper()+' (g C m-2)')
    plt.ylim((-20,15))
    if awindl!=None:
        ax.axvline(x=times[awindl],color='k',ls='dashed')
    return ax, fig


def model_dat_resid(dC, pvals, skip_yrs=1):
    """
    Calculates the mean model data difference of a forecast of nee over some time period
    :param dC: data class for DALEC2
    :param pvals: starting parameter set
    :param skip_yrs: how many years to skip when calculating mean
    :return: mean model data difference
    """
    m = mcp.DalecModel(dC)
    mod_list = m.mod_list(pvals)
    nee_lst = m.oblist('nee', mod_list)
    yhx = [nee_lst[365*x:365*x+365]-dC.obdict['nee'][365*x:365*x+365] for x in xrange(dC.endyr-dC.startyr)]
    mean_yhx = np.nanmean(yhx[skip_yrs:], axis=0)
    return mean_yhx


def model_dat_resid_plt(mean_yhx, line='o', yr=1999, figax=None):
    """
    Plots model data residual averaged over many yrs
    :param mean_yhx: mean model data residual
    :param yr: year to plot for
    :param figax: figure and ax to include
    :return: fig, ax of plot
    """
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth':2.0, 'lines.markersize':6.0})
    palette = sns.color_palette("colorblind", 11)
    if figax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig, ax = figax
    sns.set_style('ticks')
    xlist = np.arange(0, 365)
    datum = dt.datetime(int(yr), 1, 1)
    delta = dt.timedelta(hours=24)
    # Convert the time values to datetime objects
    times = []
    for t in xlist:
        times.append(datum + int(t) * delta)
    for m_yhx in enumerate(mean_yhx):
        ax.plot(times, m_yhx[1], line, color=palette[m_yhx[0]+1])
    ax.set_ylabel(r'model-data residual (g C m$^{-2}$ day$^{-1}$)')
    ax.set_xlabel('Date')
    # ax.axhline(0, color='k', linestyle='--')
    myFmt = mdates.DateFormatter('%b')  # format x-axis time to display just month
    ax.xaxis.set_major_formatter(myFmt)
    plt.ylim((-15,10))
    return fig, ax


def plotscatterobs(ob, pvals, dC, awindl, bfa='a'):
    """Plots scatter plot of obs vs model predicted values. Takes an initial
    parameter set, a dataClass (must have only desired ob for comparison
    specified in dC), assimilation window length and whether a comparison of
    background 'b', forecast 'f' or analysis 'a' is desired.
    """
    sns.set_context('poster', font_scale=1.5, rc={'lines.linewidth':1., 'lines.markersize':6.})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    sns.set_style('ticks')
    palette = sns.color_palette("colorblind", 11)
    m = mcp.DalecModel(dC)
    mod_list = m.mod_list(pvals)
    y, y_err, y_time = m.obscost()
    hx = m.hxcost(mod_list)
    split_val = 0
    for x in xrange(0,awindl):
        if np.isnan(dC.obdict[ob][x])!= True:
            split_val += 1
    oneone=np.arange(int(min(min(y),min(hx)))-1, int(max(max(y),max(hx)))+1)
    plt.plot(oneone, oneone, color=palette[0])
    if bfa == 'b' or bfa == 'a':
        ax.plot(y[0:split_val], hx[0:split_val], 'o', color=palette[1])
        error = np.sqrt(np.sum((y[0:split_val]-hx[0:split_val])**2)/\
                                                            len(y[0:split_val]))
        yhx = np.mean(y[0:split_val]-hx[0:split_val])
    elif bfa == 'f':
        ax.plot(y[split_val:], hx[split_val:], 'o', color=palette[1])
        error = np.sqrt(np.sum((y[split_val:]-hx[split_val:])**2)/\
                                                            len(y[split_val:]))
        yhx = np.mean(y[split_val:]-hx[split_val:])
    else:
        raise Exception('Please check function input for bfa variable')
    plt.xlabel(ob.upper()+r' observations (g C m$^{-2}$ day$^{-1}$)')
    plt.ylabel(ob.upper()+' model (g C m$^{-2}$ day$^{-1}$)')
    #plt.title(bfa+'_error=%f, mean(y-hx)=%f' %(error,yhx))
    print bfa+'_error=%f, mean(y-hx)=%f' % (error,yhx)
    plt.xlim((-20,15))
    plt.ylim((-20,15))
    return ax, fig #error, y[0:splitval]-hx[0:splitval]
