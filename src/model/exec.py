import modclass as mc
import numpy as np
import pickle


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