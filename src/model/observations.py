"""Possible carbon balance observations shown as functions of the DALEC
variables and parameters to be used in data assimilation scheme.
"""
import numpy as np
import ad
import model as m


def gpp(pvals, dC, x):
    """Fn calculates gross primary production (gpp).
    """
    gpp = m.acm(pvals[18], pvals[16], pvals[10], dC, x)
    return gpp


def rec(pvals, dC, x):
    """Function calculates total ecosystem respiration (rec).
    """
    rec = pvals[1]*m.acm(pvals[18], pvals[16], pvals[10], dC, x) + (pvals[7]*\
          pvals[21] +pvals[8]*pvals[22])*m.temp_term(pvals[9], dC, x)
    return rec
    

def nee(pvals, dC, x):
    """Function calculates Net Ecosystem Exchange (nee).
    """
    nee = -(1.-pvals[1])*m.acm(pvals[18], pvals[16], pvals[10], dC, x) + \
          (pvals[7]*pvals[21] + pvals[8]*pvals[22])* \
          m.temp_term(pvals[9], dC, x)
    return nee
    
    
def litresp(pvals, dC, x):
    """Function calculates litter respiration (litresp).
    """
    litresp = pvals[7]*pvals[21]*m.temp_term(pvals[9], dC, x)
    return litresp
    
    
def soilresp(pvals, dC, x):
    """Function calculates soil respiration (soilresp).
    """
    soilresp = pvals[8]*pvals[22]*m.temp_term(pvals[9], dC, x) + \
               (1./3.)*pvals[1]*m.acm(pvals[18], pvals[16], pvals[10], dC, x)
    return soilresp


def groundresp(pvals, dC, x):
    groundresp = pvals[7]*pvals[21]*m.temp_term(pvals[9], dC, x) + \
                 pvals[8]*pvals[22]*m.temp_term(pvals[9], dC, x) + \
                 (1./3.)*pvals[1]*m.acm(pvals[18], pvals[16], pvals[10], dC, x)
    return groundresp
    
    
def rh(pvals, dC, x):
    """Function calculates rh.
    """    
    rh = soilresp(pvals, dC, x) + litresp(pvals, dC, x)
    return rh
    
    
def rtot(pvals, dC, x):
    """Function calculates rtot for ground (not yet). This needs thought!!!
    """
    rtot = pvals[8]*pvals[22]*m.temp_term(pvals[9], dC, x) + 5.
    #Figure this out boi
    return rtot
    
    
def lai(pvals, dC, x):
    """Fn calculates leaf area index (cf/clma).
    """
    lai = pvals[18] / pvals[16]
    return lai
    
    
def lf(pvals, dC, x):
    """Fn calulates litter fall.
    """
    lf = m.phi_fall(pvals[14], pvals[15], pvals[4], dC, x)*pvals[18]
    return lf
    
    
def lw(pvals, dC, x):
    """Fn calulates litter fall.
    """
    lw = pvals[5]*pvals[20]
    return lw
    
    
def clab(pvals, dC, x):
    """Fn calulates labile carbon.
    """
    clab = pvals[17]
    return clab
    
    
def cf(pvals, dC, x):
    """Fn calulates foliar carbon.
    """
    cf = pvals[18]
    return cf
    
    
def cr(pvals, dC, x):
    """Fn calulates root carbon.
    """
    cr = pvals[19]
    return cr
    

def cw(pvals, dC, x):
    """Fn calulates woody biomass carbon.
    """
    cw = pvals[20]
    return cw
    
    
def cl(pvals, dC, x):
    """Fn calulates litter carbon.
    """
    cl = pvals[21]
    return cl
    
    
def cs(pvals, dC, x):
    """Fn calulates soil organic matter carbon.
    """
    cs = pvals[22]
    return cs

def d_onset(pvals, dC, x):
    """Fn calulates d_onset.
    """
    d_onset = pvals[11]
    return d_onset
    
#REDO with a list of obs as optional input!! Then ad can do rest. 
#import inspect, inspect.getargspec
def linob(ob, pvals, dC, x):
    """Function returning jacobian of observation with respect to the parameter
    list. Takes an obs string, a parameters list, a dataClass and a time step
    x.
    """
    modobdict = {'gpp': gpp, 'nee': nee, 'rt': rec, 'cf': cf, 'clab': clab, 
                 'cr': cr, 'cw': cw, 'cl': cl, 'cs': cs, 'lf': lf, 'lw': lw, 
                 'lai':lai, 'soilresp': soilresp, 'litresp': litresp,
                 'rtot': rtot, 'rh': rh, 'd_onset': d_onset}
    dpvals = ad.adnumber(pvals)
    output = modobdict[ob](dpvals, dC, x)
    return np.array(ad.jacobian(output, dpvals))