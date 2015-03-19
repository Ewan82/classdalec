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
                 endyr=None, mnth_lst=None):
        
        #Extract the data
        self.k = None
        self.obs_str = obstr
        self.homepath = os.path.expanduser("~")
        self.data = ml.csv2rec(self.homepath+\
                              "/classdalec/aliceholtdata/ahdat99_13.csv",\
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
        
        self.mnth_lst = mnth_lst

        #'I.C. for carbon pools gCm-2'   range
        self.clab = 75.0               # (10,1e3)
        self.cf = 2.0                  # (10,1e3)
        self.cr = 135.0                # (10,1e3)
        self.cw = 14313.0                # (3e3,3e4)
        self.cl = 70.                 # (10,1e3) 
        self.cs = 18624.0                # (1e3, 1e5)
        self.clist = np.array([[self.clab,self.cf,self.cr,self.cw,self.cl,\
                                self.cs]])
        
        #'Parameters for optimization'                     range
        self.p1 = 2.0e-5 #theta_min, cl to cs decomp    (1e-5 - 1e-2)day^-1
        self.p2 = 0.45 #f_auto, fraction of GPP respired  (0.3 - 0.7)
        self.p3 = 0.01 #f_fol, frac GPP to foliage        (0.01 - 0.5)
        self.p4 = 0.457 #f_roo, frac GPP to fine roots     (0.01 - 0.5)
        self.p5 = 3. #clspan, leaf lifespan             (1.0001 - 5)
        self.p6 = 0.000023 #theta_woo, wood C turnover     (2.5e-5 - 1e-3)day^-1
        self.p7 = 0.00672 #theta_roo, root C turnover rate(1e-4 - 1e-2)day^-1
        self.p8 = 0.008 #theta_lit, litter C turnover   (1e-4 - 1e-2)day^-1
        self.p9 = 0.0000026 #theta_som, SOM C turnover    (1e-7 - 1e-3)day^-1 
        self.p10 = 0.0193 #Theta, temp dependence exp fact(0.018 - 0.08)
        self.p11 = 90. #ceff, canopy efficiency param     (10 - 100)        
        self.p12 = 140. #d_onset, clab release date       (1 - 365) (60,150)
        self.p13 = 0.7 #f_lab, frac GPP to clab        (0.01 - 0.5)
        self.p14 = 27. #cronset, clab release period      (10 - 100)
        self.p15 = 308. #d_fall, date of leaf fall        (1 - 365) (242,332)
        self.p16 = 35. #crfall, leaf fall period          (10 - 100)
        self.p17 = 46.7 #clma, leaf mass per area          (10 - 400)gCm^-2 
        
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
        
        self.ahpvals = np.array([9.41000000e-04, 4.70000000e-01,2.80000000e-01,
                         2.60000000e-01,   1.01000000e+00,   2.60000000e-04,
                         2.48000000e-03,   3.38000000e-03,   2.60000000e-06,
                         1.93000000e-02,   9.00000000e+01,   1.40000000e+02,
                         4.62900000e-01,   2.70000000e+01,   3.08000000e+02,
                         3.50000000e+01,   5.20000000e+01,   78.,
                         2.,   134.,   14257.32,
                         68.95,   18625.77])
        
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
        self.acmwilliamsxls = np.array([0.0155, 1.526, 324.1, 0.2017,
                                                1.315, 2.595, 0.037, 0.2268,
                                                0.9576])
        self.acmreflex = np.array([0.0156935, 4.22273, 208.868, 0.0453194,
                                   0.37836, 7.19298, 0.011136, 2.1001, 
                                   0.789798])
        self.acm = self.acmreflex #(currently using params from REFLEX)
        self.setacm(self.acm)
        self.phi_d = -2.5 #max. soil leaf water potential difference
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
        self.year = self.fluxdata['year'] #Year
        self.month = self.fluxdata['month'] #Month
        self.date = self.fluxdata['date'] #Date in month
        
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
                                    
        self.cpoolbstnddev=np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1,
                                     0.1, 0.3, 0.3, 0.2, 0.1, 0.3, 0.1, 0.3,
                                     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  

        self.B = self.makeB(self.cpoolbstnddev)
        #MAKE NEW B, THIS IS WRONG!

        #'Observartion variances for carbon pools and NEE' 
        self.sigo_clab = (self.clab*0.1) #10%
        self.sigo_cf = (self.cf*0.1) #10%
        self.sigo_cw = (self.cw*0.1) #10%
        self.sigo_cr = (self.cr*0.3) #30%
        self.sigo_cl = (self.cl*0.3) #30%
        self.sigo_cs = (self.cs*0.3) #30% 
        self.sigo_nee = 0.71 #(gCm-2day-1)**2
        self.sigo_lf = 0.2
        self.sigo_lw = 0.2
        self.sigo_litresp = 0.6
        self.sigo_soilresp = 0.6
        self.sigo_rtot = 0.6
        self.sigo_rh = 0.6
        
        self.errdict = {'clab':self.sigo_clab, 'cf':self.sigo_cf,\
                        'cw':self.sigo_cw,'cl':self.sigo_cl,'cr':self.sigo_cr,\
                        'cs':self.sigo_cs, 'nee':self.sigo_nee,\
                        'lf':self.sigo_lf, 'lw':self.sigo_lw,\
                        'litresp':self.sigo_litresp,\
                        'soilresp':self.sigo_soilresp,\
                        'rtot':self.sigo_rtot,\
                        'rh':self.sigo_rh}
        
        if self.obs_str!=None and self.mnth_lst==None:
            self.obdict, self.oberrdict = self.assimilation_obs(self.obs_str)
        elif self.obs_str!=None and self.mnth_lst!=None:
            self.obdict, self.oberrdict = self.time_assimilation_obs(
                                                   self.obs_str, self.mnth_lst)
        else:
            self.obdict, self.oberrdict = None
            
            
    def makeB(self, bstnddevs):
        """Creates diagonal B matrix.
        """
        B = (bstnddevs*self.xb)**2*np.eye(23)
        return B
        
        
    def setacm(self, acm):
        """Sets ACM parameter values.
        """
        self.a2 = acm[0]
        self.a3 = acm[1] 
        self.a4 = acm[2] 
        self.a5 = acm[3] 
        self.a6 = acm[4] 
        self.a7 = acm[5] 
        self.a8 = acm[6] 
        self.a9 = acm[7] 
        self.a10 = acm[8]             
            
            
    def assimilation_obs(self, obs_str):
        possibleobs = ['gpp', 'lf', 'lw', 'rt', 'nee', 'cf', 'cl', \
                       'cr', 'cw', 'cs', 'lai', 'clab', 'litresp', 'soilresp',\
                       'rtot', 'rh', 'rabg']
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

        
    def time_assimilation_obs(self, obs_str, mnth_lst):
        possibleobs = ['gpp', 'lf', 'lw', 'rt', 'nee', 'cf', 'cl', \
                       'cr', 'cw', 'cs', 'lai', 'clab', 'litresp', 'soilresp',\
                       'rtot', 'rh', 'rabg']
        Obslist = re.findall(r'[^,;\s]+', obs_str)
        Obs_dict = {}
        Obs_err_dict = {}
        for x in xrange(len(Obslist)):
            if Obslist[x] not in possibleobs:
                raise Exception('Invalid observations entered, please check \
                                 function input')
            else:
                Obs_dict[Obslist[x]] = self.fluxdata[Obslist[x]]
                indlist=[]
                for t in mnth_lst:
                    indlist.append(np.where(self.month==t)[0].tolist()[:])
                indlist=[item for sublist in indlist for item in sublist]
                for t in xrange(self.lenrun):
                    if t in indlist:
                        continue
                    else:
                        Obs_dict[Obslist[x]][t] = float('NaN')
                    
                Obs_err_dict[Obslist[x]+'_err'] = (Obs_dict[Obslist[x]] / \
                                         Obs_dict[Obslist[x]])* \
                                                      self.errdict[Obslist[x]]
        
        return Obs_dict, Obs_err_dict

        
    def pickle(self, filename):
            f = open(filename, 'w')
            pickle.dump(self, f)
            f.close()
            
    