"""dataClass extracting driving data from young ponderosa Oregon pine forest
required to run dalecv2 model forwards. As a class function will also extract 
observations used in assimilation.
"""
import numpy as np
import random
import os
import re
import collections as col
import oregondata as ogd
import ahdata2 as ahd
import model as mod
import observations as obs

class dalecData( ): 

    def __init__(self, lenrun, obs_str=None, no_obs=0, freq_obs=0, startrun=0,
                 k=1, erron=1, errs='normal', errors=1):
        
        self.no_obs = no_obs
        self.obs_str = obs_str
        self.freq_obs = freq_obs
        self.lenrun = lenrun
        self.startrun = startrun
        self.timestep = np.arange(startrun, startrun+lenrun)
        self.d = ahd.DalecData(self.lenrun)
        self.erron = erron    
        self.k = k
        
        #Extract the data
        self.homepath = os.path.expanduser("~")
        self.f = open(self.homepath+"/projects/classdalec/oregondata/dalec_drivers.txt",\
                      "r")
        self.allLines = self.f.readlines()
        self.data = np.array([[-9999.]*9 for i in range(self.lenrun)])
        n = -1
        for x in xrange(self.startrun, self.lenrun+self.startrun):
            n = n + 1
            allVars = self.allLines[x].split()
            for i in xrange(0, 9):
                self.data[n,i] = float(allVars[i])

        
        #'I.C. for carbon pools gCm-2'
        self.clab = 4.78766535e+01
        self.cf = 3.27899589e+02
        self.cr = 1.69977954e+02
        self.cw = 6.40102687e+03
        self.cl = 3.63058972e+02
        self.cs = 1.01814051e+04
        self.clist = np.array([[self.clab,self.cf,self.cr,self.cw,self.cl,\
                                self.cs]])

        #'Parameters for optimization'                     range
        self.p1 = 0.0000441 #theta_min, cl to cs decomp  (1e-2 - 1e-5)day^-1
        self.p2 = 0.47 #f_auto, fraction of GPP respired  (0.3 - 0.7)
        self.p3 = 0.28 #f_fol, frac GPP to foliage        (0.01 - 0.5)
        self.p4 = 0.16 #f_roo, frac GPP to fine roots     (0.01 - 0.5)
        self.p5 = 1.5 #clspan, leaf lifespan              (? - ?)
        self.p6 = 0.000036 #theta_woo, wood C turnover  (2.5e-5 - 1e-3)day^-1
        self.p7 = 0.00248 #theta_roo, root C turnover rate(1e-4 - 1e-2)day^-1
        self.p8 = 0.00228 #theta_lit, litter C turnover    (1e-4 - 1e-2)day^-1
        self.p9 = 0.0000026 #theta_som, SOM C turnover    (1e-7 - 1e-3)day^-1 
        self.p10 = 0.0693 #Theta, temp dependence exp fact(0.018 - 0.08)
        self.p11 = 15. #ceff, canopy efficiency param     (10 - 100)        
        self.p12 = 40.4 #d_onset, clab release date       (1 - 365)
        self.p13 = 0.050629 #f_lab, frac GPP to clab      (0.01 - 0.5)
        self.p14 = 30. #cronset, clab release period      (10 - 100)
        self.p15 = 197. #d_fall, date of leaf fall        (1 - 365)
        self.p16 = 90. #crfall, leaf fall period         (10 - 100)
        self.p17 = 52. #clma, leaf mass per area          (10 - 400)gCm^-2
  
        self.paramdict = col.OrderedDict([('theta_min', self.p1), 
                       ('f_auto', self.p2), ('f_fol', self.p3), 
                       ('f_roo', self.p4), ('clspan', self.p5), 
                       ('theta_woo', self.p6), ('theta_roo', self.p7), 
                       ('theta_lit', self.p8), ('theta_som', self.p9), 
                       ('Theta', self.p10), ('ceff', self.p11), 
                       ('d_onset', self.p12), ('f_lab', self.p13), 
                       ('cronset', self.p14), ('d_fall', self.p15), 
                       ('crfall', self.p16), ('clma', self.p17),
                       ('clab', self.clab), ('cf', self.cf), 
                       ('cr', self.cr), ('cw', self.cw), ('cl', self.cl),
                       ('cs', self.cs)])
        self.pvals = np.array(self.paramdict.values())

        self.x_truth = np.array([  1.75177537e-03,   4.39216724e-01,   1.45275772e-01,
                                4.85994115e-01,   1.34919734e+00,   1.43577821e-04,
                                5.33065591e-03,   1.42217674e-03,   2.81251902e-04,
                                2.23716106e-02,   4.91216576e+01,   1.15606853e+02,
                                3.09940679e-01,   4.95177377e+01,   2.54598514e+02,
                                9.74817836e+01,   8.05741919e+01,   1.92147202e+02,
                                1.09382538e+02,   3.27096649e+02,   8.91617573e+03,
                                2.40016633e+02,   2.36359753e+03]) #from generated EDC ensem

        self.x_guess = np.array([  2.87887370e-03,   5.27924849e-01,   2.01393985e-01,
                                4.03067711e-01,   1.23305582e+00,   2.11375971e-04,
                                4.22635967e-03,   2.35355321e-03,   8.90362689e-05,
                                5.24112200e-02,   7.92041640e+01,   1.17878177e+02,
                                3.53102244e-01,   4.00692448e+01,   2.64689459e+02,
                                1.36275240e+02,   1.65420736e+02,   1.89494364e+02,
                                6.14492054e+01,   3.00083726e+02,   1.32900072e+04,
                                3.76682105e+02,   2.57863745e+03])
        
        self.pvalguess = np.array([5.42167609e-03, 3.75592769e-01, 
                             1.13412142e-01, 4.71855328e-01, 3.36472138e+00,  
                             3.60664084e-04, 1.61170462e-04, 9.66270368e-03,   
                             4.67947900e-06, 7.52885101e-02, 4.65892078e+01,   
                             1.06244959e+02, 1.30332784e-01, 7.97020647e+01,   
                             1.34175195e+02, 9.32449251e+01, 6.89688676e+01,   
                             6.95997445e+02, 2.02764409e+02, 1.63764989e+02,   
                             3.51260333e+04, 5.32639251e+02, 1.86164131e+05])
                             
                             
        self.pvalpert = np.array([3.97059562e-05, 5.50344782e-01, 
                        2.58320225e-01, 1.73508624e-01,   1.31642625e+00,   
                        3.96231993e-05, 2.22274126e-03,   1.88368710e-03,  
                        2.48436090e-06, 5.67706350e-02,   1.30009851e+01,  
                        4.82140104e+01, 4.07493287e-02,   2.71111848e+01,   
                        1.60167154e+02, 7.63930119e+01,   5.00036238e+01,   
                        3.60836639e+01, 5.13637387e+01,   1.04238366e+02,  
                        6.16257431e+02, 3.27147653e+01,   9.55209955e+03])
                        
        self.pvalburn = np.array([4.41000000e-05, 4.70000000e-01,   
                          2.80000000e-01, 1.60000000e-01, 1.50000000e+00,   
                          3.60000000e-05, 2.48000000e-03, 2.28000000e-03, 
                          2.60000000e-06, 6.93000000e-02, 1.50000000e+01, 
                          4.04000000e+01, 5.06290000e-02, 3.00000000e+01, 
                          1.97000000e+02, 9.00000000e+01, 5.20000000e+01,
                          4.78766535e+01, 3.27899589e+02, 1.69977954e+02,
                          6.40102687e+03, 3.63058972e+02, 1.01814051e+04])

        self.pvalburnpert = np.array([6.46878858e-05, 3.97543192e-01,
                               2.99653443e-01, 1.86748822e-01, 2.22597454e+00,  
                               2.67032746e-05, 2.01510400e-03, 3.32548355e-03,   
                               1.43790387e-06, 7.28409608e-02, 2.19469644e+01,  
                               4.39860970e+01, 3.24564986e-02, 4.48153570e+01,  
                               2.31502677e+02, 9.32789998e+01, 3.76239946e+01,  
                               4.32147084e+01, 4.76802246e+02, 2.34676150e+02,  
                               4.87619657e+03, 2.93150254e+02, 1.03952316e+04])
                               
        self.pvalburnpert2= np.array([5.51868034e-05, 4.37632826e-01, 
                        3.89079269e-01, 1.85104101e-01,   2.24821975e+00,  
                        6.98021967e-05, 4.57527399e-04,   2.47798868e-03,   
                        1.96309016e-06, 7.98945410e-02,   1.86092290e+01,   
                        3.23004477e+01, 1.56323093e-02,   3.26731984e+01,  
                        1.19856967e+02, 9.99073805e+01,   1.00072803e+01,  
                        7.17403552e+01, 4.57999477e+02,   1.60424961e+02, 
                        1.15858201e+04, 5.90675123e+02,   1.24358017e+04])
                        
        self.xb = self.pvalburnpert


       
        self.bnds=((1e-5,1e-2),(0.3,0.7),(0.01,0.5),(0.01,0.5),(1.0001,10.),\
              (2.5e-5,1e-3),(1e-4,1e-2),(1e-4,1e-2),(1e-7,1e-3),(0.018,0.08),\
              (10,100),(1,365),(0.01,0.5),(10,100),(1,365),(10,100),(10,400),\
              (10,1000),(10,1000),(10,1000),(100,1e5),(10,1000),(100,2e5))
              
        self.positivebnds=((0,None),(0,None),(0,None),(0,None),(0,None),\
                   (0,None),(0,None),(0,None),(0,None),(0,None),(1.0001,None),\
                   (0,None),(0,None),(0,None),(0,None),(0,None),(0,None),\
                   (0,None),(0,None),(0,None),(0,None),(0,None),(0,None))

        self.bnds2 = ((1e-5, 1e-2), (0.3, 0.7), (0.01, 0.5), (0.01, 0.5), (1.0001, 10.),
                     (2.5e-5, 1e-3), (1e-4, 1e-2), (1e-4, 1e-2), (1e-7, 1e-3), (0.018, 0.08),
                     (10, 100), (60, 150), (0.01, 0.5), (10, 100), (242, 332), (10, 150), (10, 400),
                     (10, 1000), (10, 1000), (10, 1000), (100, 1e5), (10, 1000), (100, 2e5))
  
            
        #Constants for ACM model 
        self.acmwilliamsspreadsheet = np.array([0.0155, 1.526, 324.1, 0.2017,
                                                1.315, 2.595, 0.037, 0.2268,
                                                0.9576])
        self.acmreflex = np.array([0.0156935, 4.22273, 208.868, 0.0453194,
                                   0.37836, 7.19298, 0.011136, 2.1001, 
                                   0.789798])
        self.acm = self.acmreflex #(currently using params from REFLEX)
        self.a2 = self.acm[0]
        self.a3 = self.acm[1] 
        self.a4 = self.acm[2] 
        self.a5 = self.acm[3] 
        self.a6 = self.acm[4] 
        self.a7 = self.acm[5] 
        self.a8 = self.acm[6] 
        self.a9 = self.acm[7] 
        self.a10 = self.acm[8] 
        self.phi_d = -2. #max. soil leaf water potential difference
        self.R_tot = 1. #total plant-soil hydrolic resistance
        self.lat = 0.908 #latitutde of forest site in radians
  
      
        #'Daily temperatures degC'
        self.t_mean = self.data[:,1].tolist()*k
        self.t_max = self.data[:,2].tolist()*k
        self.t_min = self.data[:,3].tolist()*k
        self.t_range = np.array(self.t_max) - np.array(self.t_min)

        
        #'Driving Data'
        self.I = self.data[:,4].tolist()*k #incident radiation
        self.ca = 355.0 #atmospheric carbon    
        self.D = self.data[:,0].tolist()*k #day of year 
 
       
        #misc
        self.radconv = 365.25 / np.pi
        
        
        #Model values
        self.pvallist = mod.mod_list(self.x_truth, self.d, 0, k*self.lenrun-1)
  
      
        #'Background variances for carbon pools & B matrix'
        self.sigb_clab = 8.36**2 #(self.clab*0.2)**2 #20%
        self.sigb_cf = 11.6**2 #(self.cf*0.2)**2 #20%
        self.sigb_cw = 20.4**2 #(self.cw*0.2)**2 #20%
        self.sigb_cr = 154.**2 #(self.cr*0.2)**2 #20%
        self.sigb_cl = 8.**2 #(self.cl*0.2)**2 #20%
        self.sigb_cs = 1979.4**2 #(self.cs*0.2)**2 #20% 
        self.B = (0.5*np.array([self.pvalburn]))**2*np.eye(23)
        #MAKE NEW B, THIS IS WRONG!
  
      
        #'Observartion variances for carbon pools and NEE'
        self.vars = np.array([self.clab*0.1, self.cf*0.1, 
                              self.cw*0.1, self.cr*0.1, 
                              self.cl*0.1, self.cs*0.1, 0.5, 0.2, 
                              0.2, 0.2, 0.4, 0.12, 0.5, 0.5])
        self.smallvars = self.vars*1e-3
        
        if errs=='normal':
            self.er = self.vars
        else:
            self.er = self.smallvars
        
        self.errdict = {'clab': self.er[0], 'cf': self.er[1],\
                        'cw': self.er[2],'cl': self.er[3],'cr': self.er[4],\
                        'cs': self.er[5], 'nee': self.er[6],\
                        'lf': self.er[7], 'lw': self.er[8], 'gpp': self.er[9],
                        'rt': self.er[10], 'lai': self.er[11],
                        'soilresp': self.er[12], 'litresp': self.er[13]}
                        
        self.modobdict = {'gpp': obs.gpp, 'nee': obs.nee, 'rt': obs.rec, 
                          'cf': obs.cf, 'clab': obs.clab, 'cr': obs.cr, 
                          'cw': obs.cw, 'cl': obs.cl, 'cs': obs.cs, 
                          'lf': obs.lf, 'lw': obs.lw, 'lai': obs.lai,
                          'soilresp': obs.soilresp, 'litresp': obs.litresp}

        if self.obs_str!=None and errors==1:
            self.obdict, self.oberrdict = self.rand_err_assim_obs(self.obs_str,
                                                              self.no_obs)
        elif self.obs_str!=None and errors==0:
            self.obdict, self.oberrdict = self.assimilation_obs(self.obs_str)
        else:
            self.obdict, self.oberrdict = None, None


    def assimilation_obs(self, obs_str):
        """Creates dictionary of synthetic obs given a string of observations.
        """
        possibleobs = ['gpp', 'lf', 'lw', 'rt', 'nee', 'cf', 'cl', \
                       'cr', 'cw', 'cs', 'lai', 'clab', 'soilresp', 'litresp']
        Obslist = re.findall(r'[^,;\s]+', obs_str)
    
        for ob in Obslist:
            if ob not in possibleobs:
                raise Exception('Invalid observations entered, please check \
                                 function input')

        Obs_dict = {ob:np.ones(self.lenrun*self.k)*float('NaN') for ob in \
                    Obslist}
        Obs_err_dict = {ob+'_err':np.ones(self.lenrun*self.k)*float('NaN') \
                        for ob in Obslist}
  
        for x in xrange(0, self.lenrun*self.k, self.freq_obs):
            for ob in Obslist:
                Obs_dict[ob][x] = self.modobdict[ob](self.pvallist[x], self.d,
                                                     x)
                Obs_err_dict[ob+'_err'][x] = self.errdict[ob]
                        
        return Obs_dict, Obs_err_dict
        
        
    def rand_err_assim_obs(self, obs_str, freq_list):
        """Creates dictionary of synthetic obs given a string of observations,
        'obs_str', and a list of number of obs, 'freq_list'.
        'freq_list' can have two forms:
        1. a lst of integers corresponding to the number of observations 
        randomly taken over the window.
        2. a list of tuples with three values, the beginning and end points 
        where obs are to be taken and the number of obs in that range.
        """
        possibleobs = ['gpp', 'lf', 'lw', 'rt', 'nee', 'cf', 'cl', \
                       'cr', 'cw', 'cs', 'lai', 'clab', 'soilresp', 'litresp']
        Obslist = re.findall(r'[^,;\s]+', obs_str)
        
        for ob in Obslist:
            if ob not in possibleobs:
                raise Exception('Invalid observations entered, please check \
                                 function input')
                                 
        Obs_dict = {ob:np.ones(self.lenrun*self.k)*float('NaN') for ob in \
                    Obslist}
        Obs_err_dict = {ob+'_err':np.ones(self.lenrun*self.k)*float('NaN') \
                        for ob in Obslist}
        if type(freq_list[0])==int:
            Obs_freq_dict = {Obslist[x]+'_freq': \
                         random.sample(range(self.lenrun*self.k), freq_list[x]) \
                         for x in xrange(len(Obslist))}
        else:
            Obs_freq_dict = {Obslist[x]+'_freq': \
                           random.sample(range(freq_list[x][0],freq_list[x][1]), 
                                       freq_list[x][2]) for x in \
                           xrange(len(Obslist))}       
                         
        for x in xrange(self.lenrun*self.k):
            for ob in Obslist:
                if x in Obs_freq_dict[ob+'_freq']:
                    if self.erron==1:
                        Obs_dict[ob][x] = self.modobdict[ob](self.pvallist[x], 
                                                         self.d, x) + \
                                      random.gauss(0, self.errdict[ob])
                    else:
                        Obs_dict[ob][x] = self.modobdict[ob](self.pvallist[x], 
                                                         self.d, x)
                    Obs_err_dict[ob+'_err'][x] = self.errdict[ob]  
                else:
                    continue
                         
        return Obs_dict, Obs_err_dict
    
    
    def randompvals(self):
        """Creates a random list of parameter values for dalec using dataClass
        bounds.
        """
        rndpvals = np.ones(23)*-9999.
        x=0
        for bnd in self.bnds:
            rndpvals[x] = np.random.uniform(bnd[0],bnd[1])
            x += 1
            
        return rndpvals

            
    def randompert(self, pvals):
        """Randomly perturbs a given list of values.
        """
        pvalapprox = np.ones(23)*-9999.
        x=0
        for p in pvals:
            pvalapprox[x] = p + random.gauss(0, p*0.5)
            if self.bnds[x][1]<pvalapprox[x]: 
                pvalapprox[x] = self.bnds[x][1] - abs(random.gauss(0, 
                                                        self.bnds[x][1]*0.001))  
            elif self.bnds[x][0]>pvalapprox[x]:
                pvalapprox[x] = self.bnds[x][0] + abs(random.gauss(0, 
                                                        self.bnds[x][0]*0.001))                
            x += 1
                 
        return pvalapprox
        
        
    def tstpvals(self, pvals):
        """Tests pvals to see if they are within the correct bnds.
        """
        x=0
        for bnd in self.bnds:
            if bnd[0]<pvals[x]<bnd[1]:
                print '%x in bnds' %x
            else:
                print '%x not in bnds' %x
            x += 1
        return pvals
        