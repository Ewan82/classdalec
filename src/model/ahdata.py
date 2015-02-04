"""dataClass extracting driving data from Alice Holt mainly deciduous forest
required to run dalecv2 model forwards. As a class function will also extract 
observations used in assimilation.
"""
import numpy as np
import os
import re
import matplotlib.mlab as ml
import collections as col
import pickle


class dalecData( ): 

    def __init__(self, lenrun=None, obstr=None, startrun=0, startyr=None, 
                 endyr=None):
        
        #Extract the data
        self.k = None
        self.obs_str = obstr
        self.homepath = os.path.expanduser("~")
        self.data = ml.csv2rec(self.homepath+\
                              "/classdalec/aliceholtdata/ahdat9905.csv",\
                              missing='nan')
        if lenrun!=None:                      
            self.lenrun = lenrun
            self.startrun = startrun
            self.fluxdata = self.data[startrun:startrun+self.lenrun]
            self.timestep = np.arange(startrun, startrun+self.lenrun)
        elif startyr!=None:
            self.fluxdata = self.data[(self.data['year']>=startyr) & \
                                      (self.data['year']< endyr)]   
            self.lenrun = len(self.fluxdata)
            self.startrun = 0   
            self.timestep = np.arange(startrun, startrun+self.lenrun)
        else:
            raise Exception('No input entered, please check function input')

        #'I.C. for carbon pools gCm-2'   range
        self.clab = 270.8               # (10,1e3)
        self.cf = 24.0                  # (10,1e3)
        self.cr = 102.0                 # (10,1e3)
        self.cw = 8100.0                # (3e3,3e4)
        self.cl = 300.0                 # (10,1e3) 
        self.cs = 7200.0                # (1e3, 1e5)
        self.clist = np.array([[self.clab,self.cf,self.cr,self.cw,self.cl,\
                                self.cs]])
        
        #'Parameters for optimization'                     range
        self.p1 = 0.000941 #theta_min, cl to cs decomp    (1e-5 - 1e-2)day^-1
        self.p2 = 0.47 #f_auto, fraction of GPP respired  (0.3 - 0.7)
        self.p3 = 0.28 #f_fol, frac GPP to foliage        (0.01 - 0.5)
        self.p4 = 0.26 #f_roo, frac GPP to fine roots     (0.01 - 0.5)
        self.p5 = 1.01 #clspan, leaf lifespan             (1.0001 - 5)
        self.p6 = 0.00026 #theta_woo, wood C turnover     (2.5e-5 - 1e-3)day^-1
        self.p7 = 0.00248 #theta_roo, root C turnover rate(1e-4 - 1e-2)day^-1
        self.p8 = 0.00338 #theta_lit, litter C turnover   (1e-4 - 1e-2)day^-1
        self.p9 = 0.0000026 #theta_som, SOM C turnover    (1e-7 - 1e-3)day^-1 
        self.p10 = 0.0193 #Theta, temp dependence exp fact(0.018 - 0.08)
        self.p11 = 90. #ceff, canopy efficiency param     (10 - 100)        
        self.p12 = 140. #d_onset, clab release date       (1 - 365) (60,150)
        self.p13 = 0.4629 #f_lab, frac GPP to clab        (0.01 - 0.5)
        self.p14 = 27. #cronset, clab release period      (10 - 100)
        self.p15 = 308. #d_fall, date of leaf fall        (1 - 365) (242,332)
        self.p16 = 35. #crfall, leaf fall period          (10 - 100)
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
        
        self.edinburghpvals=np.array([0.000189966332469257,0.565343476756027,
             0.015313852599075,0.229473358726997,1.3820788381002,
             2.56606744808776e-05,0.000653099081656547,0.00635847131570823,
             4.32163613374937e-05,0.0627274280370167,66.4118798958804,
             122.361932206327,0.372217324163812,114.092521668926,
             308.106881011017,63.6023224321684,201.056970845445,
             201.27512854457,98.9874539256948,443.230119619488,
             20293.9092250464,141.405866537237,2487.84616355469])
             
        self.xb = self.pvals
        
        self.bnds=((1e-5,1e-2),(0.3,0.7),(0.01,0.5),(0.01,0.5),(1.0001,10.),\
              (2.5e-5,1e-3),(1e-4,1e-2),(1e-4,1e-2),(1e-7,1e-3),(0.018,0.08),\
              (10,100),(1,365),(0.01,0.5),(10,100),(1,365),(10,100),(10,400),\
              (10,1000),(10,1000),(10,1000),(100,1e5),(10,1000),(100,2e5))
              
              
        self.xa = None
        
      
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
        self.lat = 0.89133965 #latitutde of forest site in radians

        
        #'Daily temperatures degC'
        self.t_mean = self.fluxdata['t_mean']
        self.t_max = self.fluxdata['t_max']
        self.t_min = self.fluxdata['t_min']
        self.t_range = np.array(self.t_max) - np.array(self.t_min)
        
        #'Driving Data'
        self.I = self.fluxdata['i'] #incident radiation
        self.ca = 390.0 #atmospheric carbon    
        self.D = self.fluxdata['day'] #day of year 
        
        #misc
        self.radconv = 365.25 / np.pi
        
        #'Background variances for carbon pools & B matrix'
        self.sigb_clab = 8.36**2 #(self.clab*0.2)**2 #20%
        self.sigb_cf = 11.6**2 #(self.cf*0.2)**2 #20%
        self.sigb_cw = 20.4**2 #(self.cw*0.2)**2 #20%
        self.sigb_cr = 154.**2 #(self.cr*0.2)**2 #20%
        self.sigb_cl = 8.**2 #(self.cl*0.2)**2 #20%
        self.sigb_cs = 1979.4**2 #(self.cs*0.2)**2 #20% 
        self.backgstnddev=np.array([0.6, 0.5, 0.5, 0.5, 0.4, 0.6, 0.6, 0.6,
                                    0.6, 0.5, 0.5, 0.2, 0.5, 0.4, 0.1, 0.4, 
                                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        self.B = (self.backgstnddev*self.xb)**2*np.eye(23)
        #MAKE NEW B, THIS IS WRONG!

        #'Observartion variances for carbon pools and NEE' 
        self.sigo_clab = (self.clab*0.1)**2 #10%
        self.sigo_cf = (self.cf*0.1)**2 #10%
        self.sigo_cw = (self.cw*0.1)**2 #10%
        self.sigo_cr = (self.cr*0.3)**2 #30%
        self.sigo_cl = (self.cl*0.3)**2 #30%
        self.sigo_cs = (self.cs*0.3)**2 #30% 
        self.sigo_nee = 0.5 #(gCm-2day-1)**2
        self.sigo_lf = 0.2**2
        self.sigo_lw = 0.2**2
        
        self.errdict = {'clab':self.sigo_clab, 'cf':self.sigo_cf,\
                        'cw':self.sigo_cw,'cl':self.sigo_cl,'cr':self.sigo_cr,\
                        'cs':self.sigo_cs, 'nee':self.sigo_nee,\
                        'lf':self.sigo_lf, 'lw':self.sigo_lw}
        
        if self.obs_str!=None:
            self.obdict, self.oberrdict = self.assimilation_obs(self.obs_str)
        else:
            self.obdict, self.oberrdict = None
            
            
    def assimilation_obs(self, obs_str):
        possibleobs = ['gpp', 'lf', 'lw', 'rt', 'nee', 'cf', 'cl', \
                       'cr', 'cw', 'cs', 'lai', 'clab']
        Obslist = re.findall(r'[^,;\s]+', obs_str)
        Obs_dict = {}
        Obs_err_dict = {}
        for ob in Obslist:
            if ob not in possibleobs:
                raise Exception('Invalid observations entered, please check \
                                 function input')
            else:
                Obs_dict[ob] = self.fluxdata[ob]
                Obs_err_dict[ob+'_err'] = (self.fluxdata[ob]/self.fluxdata[ob])\
                                          *self.errdict[ob]
        
        return Obs_dict, Obs_err_dict
        
    def pickle(self, filename):
            f = open(filename, 'w')
            pickle.dump(self, f)
            f.close()
            
    