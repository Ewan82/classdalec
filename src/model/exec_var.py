import modclass as mc
import ahdata2 as ahd2
import numpy as np
import pickle
import scipy.stats as stats
import sympy as smp
import matplotlib.pyplot as plt
import plot as p

def fourdvar_run(d, b, r_var, pvals='mean', maxiters=3000):
    """Run 4dvar with DALEC2 using specified pickled B file and diagonal R with specified variance on diagonal.
    """
    d.B = pickle.load(open(b, 'rb'))
    m = mc.DalecModel(d)
    rmat = np.eye(len(m.rmatrix))*r_var
    m.rmatrix = rmat
    if pvals=='mean':
        pvals = d.edinburghmean
    else:
        pvals = pvals
    return m.findmintnc(pvals, maxits=maxiters)

def savefig_fourdvar(d, bname, floc=None):
    """ Runs DALEC2 fourdvar code with specified B and R variance, saves plots and pickles xa to specified directory.
    """
    d.B = pickle.load(open(bname+'.p', 'rb'))
    m = mc.DalecModel(d)
    xa = m.findmintnc(d.edinburghmedian)
    f = open(floc+bname+'_xa', 'wb')
    pickle.dump(xa, f)
    f.close()
    d2 = ahd2.DalecData(startyr=d.startyr, endyr=2014, obstr='nee')
    ax, fig = p.plot4dvarrun('nee', d.edinburghmedian, xa[0], d2, 0, len(d2.I), awindl=len(d.I))
    fig.savefig(floc+bname+'_4dvar.png', bbox_inches='tight')
    plt.close()
    ax, fig = p.plotscatterobs('nee', xa[0], d2, len(d.I), 'f')
    fig.savefig(floc+bname+'_forecast_scatter.png', bbox_inches='tight')
    plt.close()
    ax, fig = p.plotscatterobs('nee', xa[0], d2, len(d.I), 'a')
    fig.savefig(floc+bname+'_analysis_scatter.png', bbox_inches='tight')
    plt.close()
    ax,fig=p.analysischange(d.edinburghmedian, xa[0])
    fig.savefig(floc+bname+'_inc.png', bbox_inches='tight')
    plt.close()
    if bname!='bdiag':
        ax, fig = p.plotbmat(pickle.load(open(bname+'_cor.p', 'rb')))
        fig.savefig(floc+bname+'_corrmat.png', bbox_inches='tight')
        plt.close()
    return xa

def savefig_fourdvar_cvt(d, bname, floc=None):
    """ Runs DALEC2 fourdvar code with specified B and R variance, saves plots and pickles xa to specified directory.
    """
    d.B = pickle.load(open(bname+'.p', 'rb'))
    m = mc.DalecModel(d)
    xa_zvals = m.findmin_cvt(d.edinburghmedian)
    xa=(m.zvals2pvals(xa_zvals[0]), xa_zvals[1], xa_zvals[2])
    f = open(floc+bname+'_xa', 'wb')
    pickle.dump(xa, f)
    f.close()
    d2 = ahd2.DalecData(startyr=d.startyr, endyr=2014, obstr='nee')
    ax, fig = p.plot4dvarrun('nee', d.edinburghmedian, xa[0], d2, 0, len(d2.I), awindl=len(d.I))
    fig.savefig(floc+bname+'_4dvar.png', bbox_inches='tight')
    plt.close()
    ax, fig = p.plotscatterobs('nee', xa[0], d2, len(d.I), 'f')
    fig.savefig(floc+bname+'_forecast_scatter.png', bbox_inches='tight')
    plt.close()
    ax, fig = p.plotscatterobs('nee', xa[0], d2, len(d.I), 'a')
    fig.savefig(floc+bname+'_analysis_scatter.png', bbox_inches='tight')
    plt.close()
    ax,fig=p.analysischange(d.edinburghmedian, xa[0])
    fig.savefig(floc+bname+'_inc.png', bbox_inches='tight')
    plt.close()
    if bname!='bdiag':
        ax, fig = p.plotbmat(pickle.load(open(bname+'_cor.p', 'rb')))
        fig.savefig(floc+bname+'_corrmat.png', bbox_inches='tight')
        plt.close()
    return xa

def test_pvals_bnds(d, pvals):
    """Tests pvals to see if they are within the correct bnds.
    """
    x=0
    for bnd in d.bnds2:
        if bnd[0]<pvals[x]<bnd[1]:
            x+=1
        else:
            return False
    return True

def pvals_test_uc(d, pvals):
    """ Test if pvals pass constraints.
    """
    #calculate carbon fluxes
    f_auto = pvals[1]
    f_fol = (1-f_auto)*pvals[2]
    f_lab = (1-f_auto-f_fol)*pvals[12]
    f_roo = (1-f_auto-f_fol-f_lab)*pvals[3]
    f_woo = (1 - f_auto - f_fol - f_lab - f_roo)

    #universal constraint tests
    uc = [10*pvals[16] > pvals[18],
          pvals[8] < pvals[7],
          pvals[0] > pvals[8],
          pvals[5] < 1/(365.25*pvals[4]),
          pvals[6] > pvals[8]*np.exp(pvals[9]*d.t_mean[0]),
          0.2*f_roo < (f_fol+f_lab) < 5*f_roo,
          pvals[14]-pvals[11]>45]
    if all(uc) == True:
        return True
    else:
        return False

def pvals_test_edc(d, pvals):
    """ Test if pvals pass constraints.
    """
    #calculate carbon fluxes
    f_auto = pvals[1]
    f_fol = (1-f_auto)*pvals[2]
    f_lab = (1-f_auto-f_fol)*pvals[12]
    f_roo = (1-f_auto-f_fol-f_lab)*pvals[3]
    f_woo = (1 - f_auto - f_fol - f_lab - f_roo)

    #universal constraint tests
    uc = [10*pvals[16] > pvals[18],
          pvals[8] < pvals[7],
          pvals[0] > pvals[8],
          pvals[5] < 1/(365.25*pvals[4]),
          pvals[6] > pvals[8]*np.exp(pvals[9]*d.t_mean[0]),
          0.2*f_roo < (f_fol+f_lab) < 5*f_roo,
          pvals[14]-pvals[11]>45]

    if all(uc) == True:
        m = mc.DalecModel(d)
        mod_list = m.mod_list(pvals) #run model
        cf_mean = np.mean(mod_list[:,18])
        cr_mean = np.mean(mod_list[:,19])
        gpp_mean = np.mean(m.oblist('gpp', mod_list))
        t_mean = np.mean(d.t_mean)
        cs_inf = (f_woo*gpp_mean) / (pvals[8]*np.exp(pvals[9]*t_mean)) + (((f_fol + f_roo + f_lab)*pvals[0])*gpp_mean) / \
                 ((pvals[0]+pvals[7])*pvals[8]*np.exp(pvals[9]*t_mean))
        cl_inf = ((f_fol + f_roo + f_lab)*gpp_mean) / (pvals[7]*np.exp(pvals[9]*t_mean))
        cw_inf = (f_woo*gpp_mean) / pvals[5]
        cr_inf = (f_roo*gpp_mean) / pvals[6]
        inf_list = [cr_inf, cw_inf, cl_inf, cs_inf]
        if (0.2*cf_mean < cr_mean < 5*cf_mean) == False: #allocation constraint
            return False

        for x in xrange(4):
            if (pvals[x+19] / 10 < inf_list[x] < 10*pvals[x+19]) == False: #Steady state constraint
                return False

        for x in xrange(17,23): #Pool growth constraint
            if (np.mean(mod_list[-365:-1,x]) / np.mean(mod_list[0:365,x]) < 1+0.1*((d.endyr-d.startyr-1)/10)) == False:
                return False

            cpoolyr1 = np.sum(mod_list[0:365, x])
            cpoolyr2 = np.sum(mod_list[366:365*2, x])
            cpoolyr1_offset = np.sum(mod_list[1:366, x])
            cpoolyr2_offset = np.sum(mod_list[367:365*2+1, x])
            delta_c0 = (cpoolyr2 - cpoolyr1) / 365
            delta_c1 = (cpoolyr2_offset - cpoolyr1_offset) / 365
            c_decay = np.log(delta_c1 / delta_c0) #exponential decay constraint
            if (c_decay > (-np.log(2) / (365.25*3))) == False:
                return False
            else:
                return True
    else:
        return False

def create_ensemble(d):
    """ Creates an ensemble of parameter values satisfying ecological constraints.
    """
    param_ensemble = []
    failed_ensemble = []
    while len(param_ensemble) < 1000:
        pvals = np.ones(23)*9999.
        for x in xrange(23):
            pvals[x] = np.random.normal(d.edinburghmean[x], d.edinburghstdev[x])
        if test_pvals_bnds(d, pvals)== True and pvals_test_uc(d, pvals)== True:
            param_ensemble.append(pvals)
            print '%i' %len(param_ensemble)
        else:
            failed_ensemble.append(pvals)
            continue
    return param_ensemble, failed_ensemble

def create_ensemble_trunc_edc(d):
    """ Creates an ensemble of parameter values satisfying ecological constraints.
    """
    trunc_dist_dict = {}
    for x in xrange(23):
        lower = d.bnds2[x][0]
        upper = d.bnds2[x][1]
        mu = d.edinburghmean[x]
        sigma = d.edinburghstdev[x]
        a = (lower-mu) / sigma
        b = (upper-mu) / sigma
        trunc_dist_dict['p%i' %int(x)] = stats.truncnorm(a, b, loc=mu, scale=sigma)

    param_ensemble = []
    failed_ensemble = []
    while len(param_ensemble) < 1500:
        pvals = np.ones(23)*9999.
        for x in xrange(23):
            pvals[x] = trunc_dist_dict['p%i' %int(x)].rvs(1)[0]
        if pvals_test_edc(d, pvals) == True:
            param_ensemble.append(pvals)
            print '%i' %len(param_ensemble)
        else:
            failed_ensemble.append(pvals)
            continue
    return np.array(param_ensemble), failed_ensemble

def create_ensemble_trunc(d):
    """ Creates an ensemble of parameter values satisfying ecological constraints.
    """
    trunc_dist_dict = {}
    for x in xrange(23):
        lower = d.bnds2[x][0]
        upper = d.bnds2[x][1]
        mu = d.edinburghmean[x]
        sigma = d.edinburghstdev[x]
        a = (lower-mu) / sigma
        b = (upper-mu) / sigma
        trunc_dist_dict['p%i' %int(x)] = stats.truncnorm(a, b, loc=mu, scale=sigma)

    param_ensemble = []
    while len(param_ensemble) < 1500:
        pvals = np.ones(23)*9999.
        for x in xrange(23):
            pvals[x] = trunc_dist_dict['p%i' %int(x)].rvs(1)[0]
        param_ensemble.append(pvals)
    return param_ensemble

def create_ensemble_trunc_uc(d):
    """ Creates an ensemble of parameter values satisfying ecological constraints.
    """
    trunc_dist_dict = {}
    for x in xrange(23):
        lower = d.bnds2[x][0]
        upper = d.bnds2[x][1]
        mu = d.edinburghmean[x]
        sigma = d.edinburghstdev[x]
        a = (lower-mu) / sigma
        b = (upper-mu) / sigma
        trunc_dist_dict['p%i' %int(x)] = stats.truncnorm(a, b, loc=mu, scale=sigma)

    param_ensemble = []
    failed_ensemble = []
    while len(param_ensemble) < 1500:
        pvals = np.ones(23)*9999.
        for x in xrange(23):
            pvals[x] = trunc_dist_dict['p%i' %int(x)].rvs(1)[0]
        if pvals_test_uc(d, pvals) == True:
            param_ensemble.append(pvals)
            print '%i' %len(param_ensemble)
        else:
            failed_ensemble.append(pvals)
            continue
    return param_ensemble, failed_ensemble

def evolve_ensemble(d, pmat):
    """Evolves an ensemble of parameters for given dataClass.
    """
    m = mc.DalecModel(d)
    modevmat = np.ones((23, 1500))*9999.
    for x in xrange(1500):
        modevmat[:, x] = m.mod_list(pmat[x])[-1]
    return modevmat

def r_mat_corr(yerroblist, ytimestep, corr=0.3, tau=1, cut_off=4):
    """ Creates a correlated R matrix.
    """
    r_corr = np.eye(len(ytimestep))
    r_diag = (yerroblist**2)*np.eye(len(yerroblist))
    for i in xrange(len(ytimestep)):
        for j in xrange(len(ytimestep)):
            if abs(ytimestep[i]-ytimestep[j]) < cut_off:
                r_corr[i,j] = corr*np.exp(-(abs(ytimestep[i]-ytimestep[j])**2)/tau**2) \
                              + (1-corr)*smp.KroneckerDelta(ytimestep[i],ytimestep[j])
    r = np.dot(np.dot((np.sqrt(r_diag)),r_corr),np.sqrt(r_diag))
    return r_corr, r

