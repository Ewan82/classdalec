import modclass as mc
import ahdata2 as ahd2
import numpy as np
import pickle
import scipy.stats as stats
import sympy as smp
import matplotlib.pyplot as plt
import plot as p


exp_list = [('bdiag', 'None'), ('b_edc', 'None'), ('bdiag', 'r_corr_cor0.3_tau4_cutoff4_var0.5'),
            ('b_edc', 'r_corr_cor0.3_tau4_cutoff4_var0.5'), ('bdiag', 'r_corr_cor0.3_tau4_cutoff4_var0.8'),
            ('b_edc', 'r_corr_cor0.3_tau4_cutoff4_var0.8'), ('bdiag', 'r_corr_cor0.3_tau6_cutoff10_var0.5'),
            ('b_edc', 'r_corr_cor0.3_tau6_cutoff10_var0.5'), ('bdiag', 'r_corr_cor0.3_tau24_cutoff24_var0.5'),
            ('b_edc', 'r_corr_cor0.3_tau24_cutoff24_var0.5'), ('bdiag', 'r_corr_cor0.6_tau4_cutoff4_var0.5'),
            ('b_edc', 'r_corr_cor0.6_tau4_cutoff4_var0.5')]
#cor, tau, cutoff, vars
exp_list2 = [('bdiag', (0., 1., 1., 0.5)), ('b_edc', (0., 1., 1., 0.5)), ('b_uc_modev1yr', (0., 1., 1., 0.5)),
             ('bdiag', (0., 1., 1., 1.)), ('b_edc', (0., 1., 1., 1.)), ('b_uc_modev1yr', (0., 1., 1., 1.)),
             ('bdiag', (0.3, 8., 4., 0.5)), ('b_edc', (0.3, 8., 4., 0.5)), ('b_uc_modev1yr', (0.3, 8., 4., 0.5)),
             ('bdiag', (0.3, 8., 2., 0.5)), ('b_edc', (0.3, 8., 2., 0.5)), ('b_uc_modev1yr', (0.3, 8., 2., 0.5)),
             ('bdiag', (0.6, 4., 4., 0.5)), ('b_edc', (0.6, 4., 4., 0.5)), ('b_uc_modev1yr', (0.6, 4., 4., 0.5))]

def pickle_mat(matrix, mat_name):
    """ Pickles error covariance matrix.
    """
    f = open(mat_name+'.p', 'wb')
    pickle.dump(matrix, f)
    f.close()
    return 'matrix pickled!'

def fourdvar_list(d, floc, matlist, pvals='mean'):
    """ Runs over a list of cov matrices.
    """
    for x in xrange(len(matlist)):
        fourdvar_run2(d, matlist[x][0], matlist[x][1], floc, pvals)

def fourdvar_run2(d, bname, r_list, floc=None, pvals='mean', maxiters=3000, f_tol=-1):
    """Run 4dvar with DALEC2 using specified pickled B file and diagonal R with specified variance on diagonal.
    """
    d.B = pickle.load(open(bname+'.p', 'rb'))
    m = mc.DalecModel(d)
    rmat = r_mat_corr(m.yerroblist, m.ytimestep, r_list[0], r_list[1], r_list[2], r_list[3])[1]
    m.rmatrix = rmat
    if pvals=='mean':
        pvals = d.edinburghmean
    else:
        pvals = pvals

    rname = 'cor%r_tau%r_cutoff%r_var%r' %(r_list)
    xa = m.findmintnc(pvals, maxits=maxiters, f_tol=f_tol)
    f = open(floc+bname+'_'+rname+'_xa', 'wb')
    pickle.dump(xa, f)
    f.close()
    d2 = ahd2.DalecData(startyr=d.startyr, endyr=2014, obstr='nee')
    ax, fig = p.plot4dvarrun('nee', d.edinburghmedian, xa[0], d2, 0, len(d2.I), awindl=len(d.I))
    fig.savefig(floc+bname+rname+'_4dvar.png', bbox_inches='tight')
    plt.close()
    ax, fig = p.plotscatterobs('nee', xa[0], d2, len(d.I), 'f')
    fig.savefig(floc+bname+rname+'_forecast_scatter.png', bbox_inches='tight')
    plt.close()
    ax, fig = p.plotscatterobs('nee', xa[0], d2, len(d.I), 'a')
    fig.savefig(floc+bname+rname+'_analysis_scatter.png', bbox_inches='tight')
    plt.close()
    ax,fig=p.analysischange(d.edinburghmedian, xa[0])
    fig.savefig(floc+bname+rname+'_inc.png', bbox_inches='tight')
    plt.close()
    if bname!='bdiag':
        ax, fig = p.plotbmat(pickle.load(open(bname+'_cor.p', 'rb')))
        fig.savefig(floc+bname+rname+'_corrmat.png', bbox_inches='tight')
        plt.close()
    ax, fig = p.plotrmat(rmat)
    fig.savefig(floc+bname+rname+'_rmat.png', bbox_inches='tight')
    plt.close()
    return xa

def fourdvar_run(d, bname, rname='None', floc=None, pvals='mean', maxiters=3000, f_tol=-1):
    """Run 4dvar with DALEC2 using specified pickled B file and diagonal R with specified variance on diagonal.
    """
    d.B = pickle.load(open(bname+'.p', 'rb'))
    m = mc.DalecModel(d)
    if rname!='None':
        rmat = pickle.load(open(rname+'.p', 'rb'))
        m.rmatrix = rmat
    if pvals=='mean':
        pvals = d.edinburghmean
    else:
        pvals = pvals
    xa = m.findmintnc(pvals, maxits=maxiters, f_tol=f_tol)
    f = open(floc+bname+'_'+rname+'_xa', 'wb')
    pickle.dump(xa, f)
    f.close()
    d2 = ahd2.DalecData(startyr=d.startyr, endyr=2014, obstr='nee')
    ax, fig = p.plot4dvarrun('nee', d.edinburghmedian, xa[0], d2, 0, len(d2.I), awindl=len(d.I))
    fig.savefig(floc+bname+rname+'_4dvar.png', bbox_inches='tight')
    plt.close()
    ax, fig = p.plotscatterobs('nee', xa[0], d2, len(d.I), 'f')
    fig.savefig(floc+bname+rname+'_forecast_scatter.png', bbox_inches='tight')
    plt.close()
    ax, fig = p.plotscatterobs('nee', xa[0], d2, len(d.I), 'a')
    fig.savefig(floc+bname+rname+'_analysis_scatter.png', bbox_inches='tight')
    plt.close()
    ax,fig=p.analysischange(d.edinburghmedian, xa[0])
    fig.savefig(floc+bname+rname+'_inc.png', bbox_inches='tight')
    plt.close()
    if bname!='bdiag':
        ax, fig = p.plotbmat(pickle.load(open(bname+'_cor.p', 'rb')))
        fig.savefig(floc+bname+rname+'_corrmat.png', bbox_inches='tight')
        plt.close()
    if rname!='None':
        ax, fig = p.plotrmat(pickle.load(open(rname+'.p', 'rb')))
        fig.savefig(floc+bname+rname+'_rmat.png', bbox_inches='tight')
        plt.close()
    return xa


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

def create_ensemble(d, mu='edin', sigma='edin'):
    """ Creates an ensemble of parameter values satisfying ecological constraints.
    """
    param_ensemble = []
    failed_ensemble = []
    if mu == 'edin':
        mu = d.edinburghmean
    if sigma == 'edin':
        sigma = d.edinburghstdev
    while len(param_ensemble) < 1000:
        pvals = np.ones(23)*9999.
        for x in xrange(23):
            pvals[x] = np.random.normal(mu[x], sigma[x])
        if test_pvals_bnds(d, pvals)== True and pvals_test_uc(d, pvals)== True:
            param_ensemble.append(pvals)
            print '%i' %len(param_ensemble)
        else:
            failed_ensemble.append(pvals)
            continue
    return param_ensemble, failed_ensemble

def create_ensemble_trunc_edc(d, mean='edin', stdev='edin'):
    """ Creates an ensemble of parameter values satisfying ecological constraints.
    """
    trunc_dist_dict = {}
    if mean == 'edin':
        mean = d.edinburghmean
    if stdev == 'edin':
        stdev = d.edinburghstdev
    for x in xrange(23):
        lower = d.bnds2[x][0]
        upper = d.bnds2[x][1]
        mu = mean[x]
        sigma = stdev[x]
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

def create_ensemble_trunc(d, mean='edin', stdev='edin'):
    """ Creates an ensemble of parameter values satisfying ecological constraints.
    """
    trunc_dist_dict = {}
    if mean == 'edin':
        mean = d.edinburghmean
    if stdev == 'edin':
        stdev = d.edinburghstdev
    for x in xrange(23):
        lower = d.bnds2[x][0]
        upper = d.bnds2[x][1]
        mu = mean[x]
        sigma = stdev[x]
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

def create_ensemble_trunc_uc(d, mean='edin', stdev='edin'):
    """ Creates an ensemble of parameter values satisfying ecological constraints.
    """
    trunc_dist_dict = {}
    if mean == 'edin':
        mean = d.edinburghmean
    if stdev == 'edin':
        stdev = d.edinburghstdev
    for x in xrange(23):
        lower = d.bnds2[x][0]
        upper = d.bnds2[x][1]
        mu = mean[x]
        sigma = stdev[x]
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

def r_mat_corr(yerroblist, ytimestep, corr=0.3, tau=1., cut_off=4., r_var=0.5):
    """ Creates a correlated R matrix.
    """
    r_corr = np.eye(len(ytimestep)) #MAKE SURE ALL VALUES ARE FLOATS FIRST!!!!
    r_diag = (r_var)*np.eye(len(yerroblist))
    for i in xrange(len(ytimestep)):
        for j in xrange(len(ytimestep)):
            if abs(ytimestep[i]-ytimestep[j]) < cut_off:
                r_corr[i,j] = corr*np.exp(-(abs(float(ytimestep[i])-float(ytimestep[j]))**2)/float(tau)**2) \
                              + (1-corr)*smp.KroneckerDelta(ytimestep[i],ytimestep[j])
    r = np.dot(np.dot((np.sqrt(r_diag)),r_corr),np.sqrt(r_diag))
    return r_corr, r

def r_mat_soar(yerroblist, ytimestep, tau=.4, cut_off=4., r_var=0.5):
    """ Creates a correlated R matrix.
    """
    r_corr = np.eye(len(ytimestep)) #MAKE SURE ALL VALUES ARE FLOATS FIRST!!!!
    r_diag = (r_var)*np.eye(len(yerroblist))
    for i in xrange(len(ytimestep)):
        for j in xrange(len(ytimestep)):
            if abs(ytimestep[i]-ytimestep[j]) < cut_off:
                r_corr[i,j] = (1+(abs(float(ytimestep[i])-float(ytimestep[j]))/float(tau)))* \
                              np.exp(-(abs(float(ytimestep[i])-float(ytimestep[j])))/float(tau))
    r = np.dot(np.dot((np.sqrt(r_diag)),r_corr),np.sqrt(r_diag))
    return r_corr, r
