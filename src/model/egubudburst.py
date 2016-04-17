import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime as dt
import seaborn as sns
import model as m
import observations as obs


xa09_13 = np.array([2.44370553e-04, 3.00000000e-01, 2.87493549e-01,
                    5.00000000e-01, 1.08406441e+00, 2.08274423e-04,
                    5.02192467e-03, 1.99010464e-03, 1.68041112e-04,
                    8.00000000e-02, 8.99790164e+01, 1.50000000e+02,
                    1.72873618e-01, 2.28636672e+01, 3.13975427e+02,
                    1.00000000e+02, 1.45960482e+02, 1.03471618e+02,
                    7.35619900e+01, 7.41470449e+02, 1.00000000e+02,
                    3.26017739e+02, 1.00000000e+02])


budburst09_13 = [111., 117., 105., 117., 123.]


def run_yearly_dalec(dC, p_lst):
    """
    Runs DALEC with a different starting parameter set for each year.
    :param dC: dataClass with driving data
    :param p_lst: list of parameter sets for each year of run
    :return: list of model evolved parameter and state values
    """
    year_lst = np.unique(dC.year)
    pval_lst = np.ones((len(dC.year), 23))
    for year in enumerate(year_lst):
        year_idx = np.where(dC.year==year[1])[0]
        pval_lst[year_idx[0]:year_idx[-1]+1]=m.mod_list(p_lst[year[0]], dC, year_idx[0], year_idx[-1])
    return pval_lst


def calc_p_lst_budburst(dC, pvals, budburst_lst):
    """
    Creates a list of starting parameter sets for each year in the dataClass (dC)
    changes date of budburst to that observed by phenocam
    :param dC: dataClass with driving data
    :param pvals: initial parameter set taken from DA run
    :param budburst_lst: list of budburst DOYs from phenocam
    :return: list of yearly starting parameter sets
    """
    year_lst = np.delete(np.unique(dC.year), -1)
    p_lst = [pvals]
    for year in enumerate(year_lst):
        year_idx = np.where(dC.year == year[1])[0]
        p = p_lst[year[0]]
        p[11] = budburst_lst[year[0]]
        p_new = m.mod_list(p, dC, year_idx[0], year_idx[-1]+1)[-1]
        p_new[11] = budburst_lst[year[0]+1]
        p_lst.append(p_new)
    return p_lst


def calc_ob_lst(dC, ob_str, pval_lst):
    """
    Creates list of observations
    :param dC: dataClass with driving data for model
    :param ob_str: sting of observation to calculate
    :param pval_lst: list of parameter values for everyday of the year for
    the run.
    :return: list of observation values.
    """
    modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 'cf': obs.cf,
                 'clab': obs.clab, 'cr': obs.cr, 'cw': obs.cw, 'cl': obs.cl,
                 'cs': obs.cs, 'lf': obs.lf, 'lw': obs.lw, 'lai':obs.lai,
                 'soilresp': obs.soilresp, 'litresp': obs.litresp,
                 'rtot': obs.rtot, 'rh': obs.rh}
    oblist = np.ones(len(pval_lst))*-9999.
    for x in xrange(len(pval_lst)):
        oblist[x] = modobdict[ob_str](pval_lst[x], dC, x)
    return oblist


def change_budburst_day(dC, p_lst, bburst_inc):
    p_lstcp = copy.deepcopy(p_lst)
    for p in p_lstcp:
        p[11] = p[11] + bburst_inc
        print p[11]
    pval_lst = run_yearly_dalec(dC, p_lstcp)
    oblist = calc_ob_lst(dC, 'nee', pval_lst)
    return oblist


def plot_bud_burst(dC, p_lst, bburst_inc_lst):
    sns.set_style('ticks')
    sns.set_context('poster', rc={'lines.linewidth':1., 'lines.markersize':3.8})
    fig, ax = plt.subplots(nrows=1, ncols=1,)
    xlist = np.arange(0, len(dC.year))
    datum = dt.datetime(int(dC.year[0]), 1, 1)
    delta = dt.timedelta(hours=24)
    # Convert the time values to datetime objects
    times = []
    for t in xlist:
        times.append(datum + int(t) * delta)

    obs_bburst = []
    for bb in bburst_inc_lst:
        print bb
        oblst = change_budburst_day(dC, p_lst, bb)
        obs_bburst.append(oblst)
        ax.plot(times, np.cumsum(oblst), label='bud burst: %d' %bb)

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Cumulative NEE (g C m-2)')
    plt.show()
    return obs_bburst, times, fig, ax