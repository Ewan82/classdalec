import modclass as mc
import numpy as np
import pickle
import scipy.stats as stats

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
    f_auto = pvals[1]
    f_fol = (1-f_auto)*pvals[2]
    f_lab = (1-f_auto-f_fol)*pvals[12]
    f_roo = (1-f_auto-f_fol-f_lab)*pvals[3]
    f_woo = (1 - f_auto - f_fol - f_lab - f_roo)

    uc = [10*pvals[16] > pvals[18],
          pvals[8] < pvals[7],
          pvals[0] > pvals[8],
          pvals[5] < 1/(365.25*pvals[4]),
          pvals[6] > pvals[8]*np.exp(pvals[9]*d.t_mean[0]),
          0.2*f_roo < (f_fol+f_lab) < 5*f_roo,
          pvals[14]-pvals[11]>45]

    if all(uc) == True:
        m = mc.DalecModel(d)
        mod_list = m.mod_list(pvals)
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
        if (0.2*cf_mean < cr_mean < 5*cf_mean) == False:
            return False

        for x in xrange(4):
            if (pvals[x+19] / 10 < inf_list[x] < 10*pvals[x+19]) == False:
                return False

        for x in xrange(17,23):

            if (np.mean(mod_list[-365:-1,x]) / np.mean(mod_list[0:365,x]) < 1+0.1*((d.endyr-d.startyr-1)/10)) == False:
                return False

            cpoolyr1 = np.sum(mod_list[0:365, x])
            cpoolyr2 = np.sum(mod_list[366:365*2, x])
            cpoolyr1_offset = np.sum(mod_list[1:366, x])
            cpoolyr2_offset = np.sum(mod_list[367:365*2+1, x])
            delta_c0 = (cpoolyr2 - cpoolyr1) / 365
            delta_c1 = (cpoolyr2_offset - cpoolyr1_offset) / 365
            c_decay = np.log(delta_c1 / delta_c0)
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