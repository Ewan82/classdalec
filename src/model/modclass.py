"""Dalecv2 model class takes a data class and then uses functions to run the
dalecv2 model.
"""
import numpy as np
import scipy.optimize as spop
import scipy.linalg as spl
import matplotlib.mlab as mlab
import algopy
import emcee


class DalecModel():

    def __init__(self, dataclass, timestep=0, strtrun=0):
        """dataClass and timestep at which to run the dalecv2 model.
        """        
        self.dC = dataclass
        self.x = timestep
        if self.dC.k is None:
            self.lenrun = self.dC.lenrun
        else:
            self.lenrun = self.dC.lenrun*self.dC.k 
        self.xb = self.dC.xb
        self.modcoston = True
        self.modobdict = {'gpp': self.gpp, 'nee': self.nee, 'rt': self.rec, 
                          'cf': self.cf, 'clab': self.clab, 'cr': self.cr, 
                          'cw': self.cw, 'cl': self.cl, 'cs': self.cs, 
                          'lf': self.lf, 'lw': self.lw, 'lai': self.lai,
                          'litresp': self.litresp, 'soilresp': self.soilresp,
                          'rtot': self.rtot, 'rh': self.rh}
        self.startrun = strtrun
        self.endrun = self.lenrun  
        self.yoblist, self.yerroblist = self.obscost()
        self.rmatrix = self.rmat(self.yerroblist)
        self.nume = 100


# ------------------------------------------------------------------------------
# Model functions
# ------------------------------------------------------------------------------

    @staticmethod
    def fitpolynomial(ep, multfac):
        """Polynomial used to find phi_f and phi (offset terms used in 
        phi_onset and phi_fall), given an evaluation point for the polynomial 
        and a multiplication term.
        """
        cf = [2.359978471e-05, 0.000332730053021, 0.000901865258885,
              -0.005437736864888, -0.020836027517787, 0.126972018064287,
              -0.188459767342504]
        polyval = cf[0]*ep**6 + cf[1]*ep**5 + cf[2]*ep**4 + cf[3]*ep**3 + cf[4]*ep**2 + \
            cf[5]*ep**1 + cf[6]*ep**0
        phi = polyval*multfac
        return phi

    def temp_term(self, Theta):
        """Calculates the temperature exponent factor for carbon pool 
        respiration's given a value for Theta parameter.
        """
        temp_term = np.exp(Theta*self.dC.t_mean[self.x])
        return temp_term

    def acm(self, cf, clma, ceff):
        """Aggregated canopy model (ACM) function
        ------------------------------------------
        Takes a foliar carbon (cf) value, leaf mass per area (clma) and canopy
        efficiency (ceff) and returns the estimated value for Gross Primary 
        Productivity (gpp) of the forest at that time.
        """
        L = cf / clma
        q = self.dC.a3 - self.dC.a4
        gc = (abs(self.dC.phi_d))**self.dC.a10 / \
             (0.5*self.dC.t_range[self.x] + self.dC.a6*self.dC.R_tot)
        p = ((ceff*L) / gc)*np.exp(self.dC.a8*self.dC.t_max[self.x])
        ci = 0.5*(self.dC.ca + q - p + np.sqrt((self.dC.ca + q - p)**2
                  - 4*(self.dC.ca*q - p*self.dC.a3)))
        E0 = (self.dC.a7*L**2) / (L**2 + self.dC.a9)
        delta = -23.4*np.cos((360.*(self.dC.D[self.x] + 10) / 365.) *
                             (np.pi/180.))*(np.pi/180.)
        s = 24*np.arccos((- np.tan(self.dC.lat)*np.tan(delta))) / np.pi
        if s >= 24.:
            s = 24.
        elif s <= 0.:
            s = 0.
        else:
            s = s
        gpp = (E0*self.dC.I[self.x]*gc*(self.dC.ca - ci))*(self.dC.a2*s +
                                                           self.dC.a5) / \
              (E0*self.dC.I[self.x] + gc*(self.dC.ca - ci))
        return gpp

    def phi_onset(self, d_onset, cronset):
        """Leaf onset function (controls labile to foliar carbon transfer) 
        takes d_onset value, cronset value and returns a value for phi_onset.
        """
        releasecoeff = np.sqrt(2.)*cronset / 2.
        magcoeff = (np.log(1.+1e-3) - np.log(1e-3)) / 2.
        offset = self.fitpolynomial(1+1e-3, releasecoeff)
        phi_onset = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff) * \
            np.exp(-(np.sin((self.dC.D[self.x] - d_onset + offset) /
                     self.dC.radconv)*(self.dC.radconv / releasecoeff))**2)
        return phi_onset

    def phi_fall(self, d_fall, crfall, clspan):
        """Leaf fall function (controls foliar to litter carbon transfer) takes 
        d_fall value, crfall value, clspan value and returns a value for
        phi_fall.
        """
        releasecoeff = np.sqrt(2.)*crfall / 2.
        magcoeff = (np.log(clspan) - np.log(clspan - 1.)) / 2.
        offset = self.fitpolynomial(clspan, releasecoeff)
        phi_fall = (2. / np.sqrt(np.pi))*(magcoeff / releasecoeff) * \
            np.exp(-(np.sin((self.dC.D[self.x] - d_fall + offset) /
                   self.dC.radconv)*self.dC.radconv / releasecoeff)**2)
        return phi_fall        

    def dalecv2(self, p):
        """DALECV2 carbon balance model 
        -------------------------------
        evolves carbon pools to the next time step, taking the 6 carbon pool 
        values and 17 parameters at time t and evolving them to time t+1.
        Outputs both the 6 evolved C pool values and the 17 constant parameter
        values.
        
        phi_on = phi_onset(d_onset, cronset)
        phi_off = phi_fall(d_fall, crfall, clspan)
        gpp = acm(cf, clma, ceff)
        temp = temp_term(Theta)
        
        clab2 = (1 - phi_on)*clab + (1-f_auto)*(1-f_fol)*f_lab*gpp
        cf2 = (1 - phi_off)*cf + phi_on*clab + (1-f_auto)*f_fol*gpp
        cr2 = (1 - theta_roo)*cr + (1-f_auto)*(1-f_fol)*(1-f_lab)*f_roo*gpp
        cw2 = (1 - theta_woo)*cw + (1-f_auto)*(1-f_fol)*(1-f_lab)*(1-f_roo)*gpp
        cl2 = (1-(theta_lit+theta_min)*temp)*cl + theta_roo*cr + phi_off*cf
        cs2 = (1 - theta_som*temp)*cs + theta_woo*cw + theta_min*temp*cl
        """        
        out = algopy.zeros(23, dtype=p)        
        
        phi_on = self.phi_onset(p[11], p[13])
        phi_off = self.phi_fall(p[14], p[15], p[4])
        gpp = self.acm(p[18], p[16], p[10])
        temp = self.temp_term(p[9])
        
        out[17] = (1 - phi_on)*p[17] + (1-p[1])*(1-p[2])*p[12]*gpp
        out[18] = (1 - phi_off)*p[18] + phi_on*p[17] + (1-p[1])*p[2]*gpp
        out[19] = (1 - p[6])*p[19] + (1-p[1])*(1-p[2])*(1-p[12])*p[3]*gpp
        out[20] = (1 - p[5])*p[20] + (1-p[1])*(1-p[2])*(1-p[12])*(1-p[3])*gpp
        out[21] = (1-(p[7]+p[0])*temp)*p[21] + p[6]*p[19] + phi_off*p[18]
        out[22] = (1 - p[8]*temp)*p[22] + p[5]*p[20] + p[0]*temp*p[21]
        out[0:17] = p[0:17]
        return out


    def dalecv2diff(self, p):
        """DALECV2 carbon balance model 
        -------------------------------
        evolves carbon pools to the next time step, taking the 6 carbon pool 
        values and 17 parameters at time t and evolving them to time t+1.
        Ouputs an array of just the 6 evolved C pool values.
        
        phi_on = phi_onset(d_onset, cronset)
        phi_off = phi_fall(d_fall, crfall, clspan)
        gpp = acm(cf, clma, ceff)
        temp = temp_term(Theta)
        
        clab2 = (1 - phi_on)*clab + (1-f_auto)*(1-f_fol)*f_lab*gpp
        cf2 = (1 - phi_off)*cf + phi_on*clab + (1-f_auto)*f_fol*gpp
        cr2 = (1 - theta_roo)*cr + (1-f_auto)*(1-f_fol)*(1-f_lab)*f_roo*gpp
        cw2 = (1 - theta_woo)*cw + (1-f_auto)*(1-f_fol)*(1-f_lab)*(1-f_roo)*gpp
        cl2 = (1-(theta_lit+theta_min)*temp)*cl + theta_roo*cr + phi_off*cf
        cs2 = (1 - theta_som*temp)*cs + theta_woo*cw + theta_min*temp*cl
        """        
        out = algopy.zeros(6, dtype=p)        
        
        phi_on = self.phi_onset(p[11], p[13])
        phi_off = self.phi_fall(p[14], p[15], p[4])
        gpp = self.acm(p[18], p[16], p[10])
        temp = self.temp_term(p[9])
        
        out[0] = (1 - phi_on)*p[17] + (1-p[1])*(1-p[2])*p[12]*gpp
        out[1] = (1 - phi_off)*p[18] + phi_on*p[17] + (1-p[1])*p[2]*gpp
        out[2] = (1 - p[6])*p[19] + (1-p[1])*(1-p[2])*(1-p[12])*p[3]*gpp
        out[3] = (1 - p[5])*p[20] + (1-p[1])*(1-p[2])*(1-p[12])*(1-p[3])*gpp
        out[4] = (1-(p[7]+p[0])*temp)*p[21] + p[6]*p[19] + phi_off*p[18]
        out[5] = (1 - p[8]*temp)*p[22] + p[5]*p[20] + p[0]*temp*p[21]
        return out

    def jac_dalecv2(self, p):
        """Using algopy package calculates the jacobian for dalecv2 given a 
        input vector p.
        """
        p = algopy.UTPM.init_jacobian(p)
        return algopy.UTPM.extract_jacobian(self.dalecv2(p)) 

    def jac2_dalecv2(self, p):
        """Use algopy reverse mode ad calc jac of dv2.
        """
        mat = np.ones((23, 23))*-9999.
        mat[0:17] = np.eye(17, 23)
        p = algopy.UTPM.init_jacobian(p)
        mat[17:] = algopy.UTPM.extract_jacobian(self.dalecv2diff(p))
        return mat

    def mod_list(self, pvals):
        """Creates an array of evolving model values using dalecv2 function.
        Takes a list of initial param values.
        """            
        mod_list = np.concatenate((np.array([pvals]),
                                  np.ones((self.endrun-self.startrun, len(pvals)))*-9999.))
        
        self.x = self.startrun
        for t in xrange(self.endrun-self.startrun):
            mod_list[(t+1)] = self.dalecv2(mod_list[t])
            self.x += 1
        
        self.x -= self.endrun
        return mod_list

    def linmod_list(self, pvals):
        """Creates an array of linearized models (Mi's) taking a list of 
        initial param values and a run length (lenrun).
        """           
        mod_list = np.concatenate((np.array([pvals]),
                                  np.ones((self.endrun-self.startrun, len(pvals)))*-9999.))
        matlist = np.ones((self.endrun-self.startrun, 23, 23))*-9999.
        
        self.x = self.startrun        
        for t in xrange(self.endrun-self.startrun):
            mod_list[(t+1)] = self.dalecv2(mod_list[t])
            matlist[t] = self.jac2_dalecv2(mod_list[t])
            self.x += 1
            
        self.x -= self.endrun    
        return mod_list, matlist

    @staticmethod
    def mfac(matlist, timestep):
        """matrix factorial function, takes a list of matrices and a time step,
        returns the matrix factoral.
        """
        if timestep == -1.:
            return np.eye(23)
        mat = matlist[0]
        for t in xrange(0, timestep):
            mat = np.dot(matlist[t+1], mat)
        return mat
        

# ------------------------------------------------------------------------------
# Observation functions
# ------------------------------------------------------------------------------

    def gpp(self, p):
        """Function calculates gross primary production (gpp).
        """
        gpp = self.acm(p[18], p[16], p[10])
        return gpp

    def rec(self, p):
        """Function calculates total ecosystem respiration (rec).
        """
        rec = p[1]*self.acm(p[18], p[16], p[10]) + \
            (p[7]*p[21] + p[8]*p[22])*self.temp_term(p[9])
        return rec

    def nee(self, p):
        """Function calculates Net Ecosystem Exchange (nee).
        """
        nee = -(1. - p[1])*self.acm(p[18], p[16], p[10]) + \
            (p[7]*p[21] + p[8]*p[22])*self.temp_term(p[9])
        return nee

    def litresp(self, p):
        """Function calculates litter respiration (litresp).
        """
        litresp = p[7]*p[21]*self.temp_term(p[9])
        return litresp

    def soilresp(self, p):
        """Function calculates soil respiration (soilresp). (heterotrophic)
        """
        soilresp = p[8]*p[22]*self.temp_term(p[9])
        return soilresp

    def rh(self, p):
        """Fn calculates rh (soilresp+litrep).
        """
        rh = self.litresp(p) + self.soilresp(p)
        return rh

    def rtot(self, p):
        """Function calculates soil + root respiration (soilrootresp).
        """
        rtot = p[8]*p[22]*self.temp_term(p[9]) + 5. #Figure this out boi
        return rtot

    def lai(self, p):
        """Fn calculates leaf area index (cf/clma).
        """
        lai = p[18] / p[16]
        return lai

    def lf(self, p):
        """Fn calulates litter fall.
        """
        lf = self.phi_fall(p[14], p[15], p[4])*p[18]
        return lf

    def lw(self, p):
        """Fn calulates litter fall.
        """
        lw = p[5]*p[20]
        return lw

    def clab(self, p):
        """Fn calulates labile carbon.
        """
        clab = p[17]
        return clab

    def cf(self, p):
        """Fn calulates foliar carbon.
        """
        cf = p[18]
        return cf

    def cr(self, p):
        """Fn calulates root carbon.
        """
        cr = p[19]
        return cr

    def cw(self, p):
        """Fn calulates woody biomass carbon.
        """
        cw = p[20]
        return cw

    def cl(self, p):
        """Fn calulates litter carbon.
        """
        cl = p[21]
        return cl

    def cs(self, p):
        """Fn calulates soil organic matter carbon.
        """
        cs = p[22]
        return cs

    def linob(self, ob, pvals):
        """Function returning jacobian of observation with respect to the 
        parameter list. Takes an obs string, a parameters list, a dataClass 
        and a time step x.
        """
        dpvals = algopy.UTPM.init_jacobian(pvals)
        return algopy.UTPM.extract_jacobian(self.modobdict[ob](dpvals)) 
        
        
# ------------------------------------------------------------------------------
# Assimilation functions
# ------------------------------------------------------------------------------

    def bmat(self, corr=False, varyp=False, edin=False):
        """Attempt at creating a b matrix.
        """
        pmat = np.ones((23, 1500))*9999.
        modevmat = np.ones((23, 1500))*9999.

        if edin is True:
            params = mlab.csv2rec('alice_holt_parameters.csv')
            pmat = np.array(params.tolist())

            for x in xrange(1500):
                modevmat[:, x] = self.mod_list(pmat[x][0:23])[-1]

            if corr is False:
                return np.cov(modevmat)
            elif corr is True:
                return np.corrcoef(modevmat)
        
        if varyp is False:
            for x in xrange(23):
                if x < 17.:
                    for i in xrange(1500):
                        pmat[x, i] = self.dC.pvals[x]
                elif x >= 17.:
                    pmat[x] = np.random.normal(self.dC.pvals[x], 
                                               self.dC.pvals[x]*0.3, 1500)
        else:
            for x in xrange(23):
                pmat[x] = np.random.normal(self.dC.pvals[x], 
                                           self.dC.pvals[x]*0.3, 1500)
        
        for x in xrange(1500):
            if pmat[4, x] < 1.0001:
                pmat[4, x] = 1.0001
                                       
        for x in xrange(1500):
            modevmat[:, x] = self.mod_list(pmat[:, x])[-1]
        
        if corr is False:
            return np.cov(modevmat)
        elif corr is True:
            return np.corrcoef(modevmat)

    def bmat2(self, corr=False):
        pmat = np.ones((23, 1500))*9999.
        modevmat = np.ones((23, 1500*self.lenrun))*9999.
        for x in xrange(23):
            pmat[x] = np.random.normal(self.dC.edinburghmedian[x], self.dC.edinburghstdev[x], 1500)
        for x in xrange(1500):
            if pmat[4,x] < 1.0001:
                pmat[4,x] = 1.0001
            modevmat[:, x*self.lenrun:self.lenrun*x+self.lenrun] = (self.mod_list(pmat[:,x])[0:self.lenrun]).T
        #return modevmat
        if corr==True:
            return np.corrcoef(modevmat)
        else:
            return np.cov(modevmat)


    def obscost(self):
        """Function returning list of observations and a list of their 
        corresponding error values. Takes observation dictionary and an 
        observation error dictionary.
        """
        yoblist = np.array([])
        yerrlist = np.array([])
        for t in xrange(self.startrun, self.endrun):
            for ob in self.dC.obdict.iterkeys():
                if np.isnan(self.dC.obdict[ob][t]) != True:
                    yoblist = np.append(yoblist, self.dC.obdict[ob][t])
                    yerrlist = np.append(yerrlist,
                                         self.dC.oberrdict[ob+'_err'][t])
        return yoblist, yerrlist

    def hxcost(self, pvallist):
        """Function returning a list of observation values as predicted by the 
        DALEC model. Takes a list of model values (pvallist), an observation 
        dictionary and a dataClass (dC).
        """
        hx = np.array([])
        self.x = self.startrun
        for t in xrange(self.startrun, self.endrun):
            for ob in self.dC.obdict.iterkeys():
                if np.isnan(self.dC.obdict[ob][t]) != True:
                    hx = np.append(hx,
                                   self.modobdict[ob](pvallist[t-self.startrun]))
            self.x += 1
        
        self.x -= self.endrun
        return hx

    def rmat(self, yerr):
        """Returns observation error covariance matrix given a list of 
        observation error values.
        """
        r = (yerr**2)*np.eye(len(yerr))
        return r

    def hmat(self, pvallist, matlist):
        """Returns a list of observation values as predicted by DALEC (hx) and 
        a linearzied observation error covariance matrix (hmat). Takes a list 
        of model values (pvallist), a observation dictionary, a list of
        linearized models (matlist) and a dataClass (dC).
        """
        hx = np.array([])
        hmat = np.array([])
        self.x = self.startrun
        for t in xrange(self.startrun, self.endrun):
            temp = []
            for ob in self.dC.obdict.iterkeys():
                if np.isnan(self.dC.obdict[ob][t]) != True:
                    hx = np.append(hx,
                                   self.modobdict[ob](pvallist[t-self.startrun]))
                    temp.append([self.linob(ob, pvallist[t-self.startrun])])
            self.x += 1
            if len(temp) != 0.:
                hmat = np.append(hmat, np.dot(np.vstack(temp),
                                 self.mfac(matlist, t-self.startrun-1)))
            else:
                continue
                        
        self.x -= self.endrun    
        hmat = np.reshape(hmat, (len(hmat)/23, 23))
        return hx, hmat 

    def modcost(self, pvals):
        """model part of cost fn.
        """
        return np.dot(np.dot((pvals-self.xb), np.linalg.inv(self.dC.B)), (pvals-self.xb).T)

    def obcost(self, pvals):
        """Observational part of cost fn.
        """
        pvallist = self.mod_list(pvals)
        hx = self.hxcost(pvallist)
        return np.dot(np.dot((self.yoblist-hx), np.linalg.inv(self.rmatrix)), (self.yoblist-hx).T)

    def cost(self, pvals):
        """4DVAR cost function to be minimized. Takes an initial state (pvals), 
        an observation dictionary, observation error dictionary, a dataClass 
        and a start and finish time step.
        """
        ob_cost = self.obcost(pvals)
        if self.modcoston is True:
            mod_cost = self.modcost(pvals)
        else:
            mod_cost = 0
        cost = 0.5*ob_cost + 0.5*mod_cost
        return cost

    def gradcost(self, pvals):
        """Gradient of 4DVAR cost fn to be passed to optimization routine. 
        Takes an initial state (pvals), an obs dictionary, an obs error 
        dictionary, a dataClass and a start and finish time step.
        """
        pvallist, matlist = self.linmod_list(pvals)
        hx, hmatrix = self.hmat(pvallist, matlist)
        obcost = np.dot(hmatrix.T, np.dot(np.linalg.inv(self.rmatrix),
                                          (self.yoblist-hx).T))
        if self.modcoston is True:
            modcost = np.dot(np.linalg.inv(self.dC.B), (pvals-self.xb).T)
        else:
            modcost = 0
        gradcost = - obcost + modcost
        return gradcost

    def acovmat(self, pvals):
        """Calculates approximation to analysis error covariance matrix
        A = (B^(-1) + H^(T) * R^(-1) * H)^(-1).
        """
        pvallist, matlist = self.linmod_list(pvals)
        hx, hmatrix = self.hmat(pvallist, matlist)
        return np.linalg.inv(np.linalg.inv(self.dC.B) + np.dot(hmatrix.T,
                        np.dot(np.linalg.inv(self.rmatrix), hmatrix)))

    def influence_mat(self, pvals):
        """Calculates the Influence matrix
        """
        pvallist, matlist = self.linmod_list(pvals)
        hx, hmatrix = self.hmat(pvallist, matlist)
        a_mat = np.linalg.inv(np.linalg.inv(self.dC.B) + np.dot(np.dot(
                              hmatrix.T, np.linalg.inv(self.rmatrix)), hmatrix))
        s_mat = np.dot(np.linalg.inv(self.rmatrix), np.dot(hmatrix, np.dot(
                              a_mat, hmatrix.T)))
        return s_mat

    def analytic_xa(self, pvals):
        """See how good linear approx is.
        """
        pvallist, matlist = self.linmod_list(pvals)
        hx, hmatrix = self.hmat(pvallist, matlist)
        k_mat = np.dot(np.dot(self.dC.B, hmatrix.T),
                       np.linalg.inv(np.dot(np.dot(hmatrix, self.dC.B), hmatrix.T) + self.rmatrix))
        xa = pvals + np.dot(k_mat, (self.yoblist - hx))
        return xa

# ------------------------------------------------------------------------------
# CVT and implied B.
# ------------------------------------------------------------------------------

    def modcost_cvt(self, zvals):
        """model part of cost fn.
        """
        return np.dot(zvals, zvals.T)

    def obcost_cvt(self, zvals):
        """Observational part of cost fn.
        """
        pvals = (np.dot(spl.sqrtm(self.dC.B),zvals)+self.xb)
        pvallist = self.mod_list(pvals)
        hx = self.hxcost(pvallist)
        return np.dot(np.dot((self.yoblist-hx), np.linalg.inv(self.rmatrix)), (self.yoblist-hx).T)

    def cost_cvt(self, zvals):
        """4DVAR cost function to be minimized. Takes an initial state (pvals),
        an observation dictionary, observation error dictionary, a dataClass
        and a start and finish time step.
        """
        ob_cost = self.obcost_cvt(zvals)
        if self.modcoston is True:
            mod_cost = self.modcost_cvt(zvals)
        else:
            mod_cost = 0
        cost = 0.5*ob_cost + 0.5*mod_cost
        return cost

    def gradcost_cvt(self, zvals):
        """Gradient of 4DVAR cost fn to be passed to optimization routine.
        Takes an initial state (pvals), an obs dictionary, an obs error
        dictionary, a dataClass and a start and finish time step.
        """
        pvals = (np.dot(spl.sqrtm(self.dC.B),zvals)+self.xb)
        pvallist, matlist = self.linmod_list(pvals)
        hx, hmatrix = self.hmat(pvallist, matlist)
        obcost = np.dot(spl.sqrtm(self.dC.B).T, np.dot(hmatrix.T, np.dot(np.linalg.inv(self.rmatrix),
                                          (self.yoblist-hx).T)))
        if self.modcoston is True:
            modcost = zvals.T
        else:
            modcost = 0
        gradcost = - obcost + modcost
        return gradcost

    def pvals2zvals(self, pvals):
        """Convert x_0 state to z_0 state for CVT with DALEC.
        """
        Bnegsqrt = np.linalg.inv(spl.sqrtm(self.dC.B))
        return np.dot(Bnegsqrt, (pvals - self.xb))

    def zvals2pvals(self, zvals):
        """Convert z_0 to x_0 for CVT.
        """
        return (np.dot(spl.sqrtm(self.dC.B),zvals)+self.xb)

    def zvalbnds(self, bnds):
        """Calculates bounds for transformed problem.
        """
        lower_bnds = []
        upper_bnds = []
        for t in bnds:
            lower_bnds.append(t[0])
            upper_bnds.append(t[1])
        zval_lowerbnds = self.pvals2zvals(np.array(lower_bnds))
        zval_upperbnds = self.pvals2zvals(np.array(upper_bnds))
        new_bnds=[]
        for t in xrange(len(bnds)):
            new_bnds.append((zval_lowerbnds[t],zval_upperbnds[t]))
        return tuple(new_bnds)

    def findmin_cvt(self, pvals, bnds='strict', dispp=None, maxits=2000,
                   mini=0):
        """Function which minimizes 4DVAR cost fn. Takes an initial state
        (pvals).
        """
        self.xb = pvals
        zvals = self.pvals2zvals(pvals)
        if bnds == 'strict':
            bnds = self.zvalbnds(self.dC.bnds)
        else:
            bnds = bnds
        findmin = spop.fmin_tnc(self.cost_cvt, zvals,
                                fprime=self.gradcost_cvt, bounds=bnds,
                                disp=dispp, fmin=mini, maxfun=maxits)
        return findmin

# ------------------------------------------------------------------------------
# Minimization Routines.
# ------------------------------------------------------------------------------
        
    def findmin(self, pvals, meth='L-BFGS-B', bnds='strict', factr=1e7):
        """Function which minimizes 4DVAR cost fn. Takes an initial state
        (pvals).
        """
        if bnds == 'strict':
            bnds = self.dC.bnds
        else:
            bnds = bnds
        findmin = spop.minimize(self.cost, pvals, method=meth,
                                jac=self.gradcost, bounds=bnds,
                                options={'gtol': 1e-1, 'disp': True, 'iprint': 2,
                                         'ftol': factr})
        return findmin

    def findminlbfgsb(self, pvals, bnds='strict', factr=1e7, prnt=-1):
        """Function which minimizes 4DVAR cost fn. Takes an initial state
        (pvals).
        """
        if bnds == 'strict':
            bnds = self.dC.bnds
        else:
            bnds = bnds
        findmin = spop.fmin_l_bfgs_b(self.cost, pvals,
                                     fprime=self.gradcost, bounds=bnds,
                                     iprint=prnt, factr=factr)
        return findmin             

    def findmintnc(self, pvals, bnds='strict', dispp=None, maxits=2000, 
                   mini=0):
        """Function which minimizes 4DVAR cost fn. Takes an initial state
        (pvals).
        """
        self.xb = pvals
        if bnds == 'strict':
            bnds = self.dC.bnds
        else:
            bnds = bnds
        findmin = spop.fmin_tnc(self.cost, pvals,
                                fprime=self.gradcost, bounds=bnds,
                                disp=dispp, fmin=mini, maxfun=maxits)
        return findmin        

    def findminglob(self, pvals, meth='TNC', bnds='strict', it=300,
                    stpsize=0.5, temp=1., displ=True, maxits=3000):
        """Function which minimizes 4DVAR cost fn. Takes an initial state 
        (pvals), an obs dictionary, an obs error dictionary, a dataClass and 
        a start and finish time step.
        """
        if bnds == 'strict':
            bnds = self.dC.bnds
        else:
            bnds = bnds
        findmin = spop.basinhopping(self.cost, pvals, niter=it,
                                    minimizer_kwargs={'method': meth, 'bounds': bnds,
                                                      'jac': self.gradcost,
                                                      'options': {'maxiter': maxits}},
                                    stepsize=stpsize, T=temp, disp=displ)
        return findmin

    def ensemble(self, pvals):
        """Ensemble 4DVAR run for twin experiments.
        """
        ensempvals = np.ones((self.nume, 23))
        for x in xrange(self.nume):
            ensempvals[x] = self.dC.randompert(pvals)
            
        assim_results = [self.findmintnc(ensemp, dispp=False) for ensemp in  
                         ensempvals]
        
        xalist = [assim_results[x][0] for x in xrange(self.nume)]
      
        return ensempvals, xalist, assim_results
        

# ------------------------------------------------------------------------------
# Cycled 4D-Var.
# ------------------------------------------------------------------------------
        
    def cycle4dvar(self, pvals, lenwind, numbwind, lenrun):        
        """Cycle 4Dvar windows and see their effect on predicting future obs.           
        """
        xb = [pvals]
        xa = []
        self.startrun = 0
        self.endrun = lenwind
        for x in xrange(numbwind):
            self.yoblist, self.yerroblist = self.obscost()
            self.rmatrix = self.rmat(self.yerroblist)
            xa.append(self.findmintnc(xb[x]))
            xb.append(self.mod_list(xa[x][0])[self.endrun-self.startrun])
            self.startrun += lenwind
            self.endrun += lenwind
        
        self.startrun -= lenwind*numbwind
        self.endrun -= lenwind*numbwind
        conditions = {'pvals': pvals, 'lenwind': lenwind, 'numbwind': numbwind,
                      'lenrun': lenrun}
        return conditions, xb, xa


# ------------------------------------------------------------------------------
# MCMC
# ------------------------------------------------------------------------------

    def randompvals(self, nwalkers):
        """Creates a random list of parameter values for dalec using dataClass
        bounds.
        """
        rndpvals = np.ones((nwalkers, 23))*-9999.
        x=0
        for t in xrange(nwalkers):
            for bnd in self.dC.bnds:
                rndpvals[t, x] = np.random.uniform(bnd[0],bnd[1])
                x += 1
            x-=23

        return rndpvals

    def log_prior(self, pvals):
        tick = 23
        for x in xrange(23):
            if self.dC.bnds[x][0]<pvals[x]<self.dC.bnds[x][1]:
                tick -= 1
        if tick==0:
            return 0.0
        return -np.inf

    def log_posterior(self, pvals):
        lp = self.log_prior(pvals)
        if not np.isfinite(lp):
            return -np.inf
        return self.log_prior(pvals) - self.obcost(pvals)

    def mcmc_run(self, pvals, ndim=23, nwalkers=50, nsteps=10000):
        """
        ndim = 23  # number of parameters in the model
        nwalkers = 50 # number of MCMC walkers
        nburn = 1000  # "burn-in" period to let chains stabilize
        nsteps = 10000  # number of MCMC steps to take
        """
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)
        sampler.run_mcmc(pvals, nsteps)
        return sampler

    def mean_pval(self, sampler, nburn=1000):
        """
        nburn = 1000  # "burn-in" period to let chains stabilize
        """
        sample = sampler.chain[:, nburn:, :].reshape(-1, 23)
        return sample.mean(0)