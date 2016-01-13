"""dataClass extracting driving data from young ponderosa Oregon pine forest
required to run dalecv2 model forwards. As a class function will also extract
observations used in assimilation.
"""
import numpy as np
import random
import re
import collections as col
import ahdata2 as ahd
import model as mod
import observations as obs
import matplotlib.mlab as ml

class dalecData( ):

    def __init__(self, lenrun, obs_str=None, no_obs=0, freq_obs=0, startrun=0,
                 k=1, erron=1, errs='normal', errors=1):

        # Extract the data
        self.k = None
        self.obs_str = obs_str
        self.data = ml.csv2rec("../../aliceholtdata/ahdat99_13.csv",
                               missing='nan')
        if lenrun is not None:
            self.lenrun = lenrun
            self.startrun = startrun
            self.fluxdata = self.data[startrun:startrun+self.lenrun]
            self.timestep = np.arange(startrun, startrun+self.lenrun)
        else:
            raise Exception('No input entered, please check function input')

        self.no_obs = no_obs
        self.obs_str = obs_str
        self.freq_obs = freq_obs
        self.timestep = np.arange(startrun, startrun+lenrun)
        self.d = ahd.DalecData(self.lenrun)
        self.erron = erron
        self.k = k

        # 'I.C. for carbon pools gCm-2'   range
        self.clab = 75.0               # (10,1e3)
        self.cf = 10.1                  # (10,1e3)
        self.cr = 135.0                # (10,1e3)
        self.cw = 14313.0              # (3e3,3e4)
        self.cl = 70.                  # (10,1e3)
        self.cs = 18624.0              # (1e3, 1e5)
        self.clist = np.array([[self.clab, self.cf, self.cr, self.cw, self.cl,
                                self.cs]])

        # 'Parameters for optimization'                    range
        self.p1 = 1.1e-5  # theta_min, cl to cs decomp      (1e-5 - 1e-2)day^-1
        self.p2 = 0.45  # f_auto, fraction of GPP respired  (0.3 - 0.7)
        self.p3 = 0.013  # f_fol, frac GPP to foliage       (0.01 - 0.5)
        self.p4 = 0.457  # f_roo, frac GPP to fine roots    (0.01 - 0.5)
        self.p5 = 3.  # clspan, leaf lifespan               (1.0001 - 5)
        self.p6 = 4.8e-5  # theta_woo, wood C turnover      (2.5e-5 - 1e-3)day^-1
        self.p7 = 6.72e-3  # theta_roo, root C turnover rate(1e-4 - 1e-2)day^-1
        self.p8 = 0.008  # theta_lit, litter C turnover     (1e-4 - 1e-2)day^-1
        self.p9 = 2.4e-5  # theta_som, SOM C turnover       (1e-7 - 1e-3)day^-1
        self.p10 = 0.0193  # Theta, temp dependence exp fact(0.018 - 0.08)
        self.p11 = 90.  # ceff, canopy efficiency param     (10 - 100)
        self.p12 = 140.  # d_onset, clab release date       (1 - 365) (60,150)
        self.p13 = 0.4  # f_lab, frac GPP to clab           (0.01 - 0.5)
        self.p14 = 27.  # cronset, clab release period      (10 - 100)
        self.p15 = 308.  # d_fall, date of leaf fall        (1 - 365) (242,332)
        self.p16 = 35.  # crfall, leaf fall period          (10 - 100)
        self.p17 = 24.2  # clma, leaf mass per area         (10 - 400)gCm^-2

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

        self.ahpvals = np.array([9.41000000e-04, 4.70000000e-01, 2.80000000e-01,
                                 2.60000000e-01,   1.01000000e+00,   2.60000000e-04,
                                 2.48000000e-03,   3.38000000e-03,   2.60000000e-06,
                                 1.93000000e-02,   9.00000000e+01,   1.40000000e+02,
                                 4.62900000e-01,   2.70000000e+01,   3.08000000e+02,
                                 3.50000000e+01,   5.20000000e+01,   78.,
                                 2.,   134.,   14257.32,
                                 68.95,   18625.77])

        self.edcpvals = np.array([  9.41000000e-04,   4.70000000e-01,   2.80000000e-01,
                                    2.60000000e-01,   1.20000000e+00,   2.60000000e-04,
                                    2.48000000e-03,   3.38000000e-03,   2.60000000e-06,
                                    1.93000000e-02,   9.00000000e+01,   1.40000000e+02,
                                    4.00000000e-01,   2.70000000e+01,   3.08000000e+02,
                                    3.50000000e+01,   5.20000000e+01,   8.00000000e+01,
                                    1.00000000e+01,   1.34000000e+02,   1.42570000e+04,
                                    6.90000000e+01,   1.86260000e+04])

        self.edcpvals2 = np.array([ 9.29180076e-04,   5.31804031e-01,   6.69448981e-02,
                                    4.46049258e-01,   1.18143120e+00,   5.31584216e-05,
                                    2.25487423e-04,   2.44782152e-03,   7.71092378e-05,
                                    3.82591095e-02,   7.47751776e+01,   1.36238252e+02,
                                    3.26252225e-01,   3.18554035e+01,   3.07257813e+02,
                                    1.20915004e+02,   1.00533213e+02,   1.27804720e+02,
                                    6.02259491e+01,   2.09997016e+02,   4.22672530e+03,
                                    3.67801053e+02,   1.62565304e+03])

        self.edinburghpvals = np.array([0.000189966332469257, 0.565343476756027,
                                        0.015313852599075, 0.229473358726997, 1.3820788381002,
                                        2.56606744808776e-05, 0.000653099081656547, 0.00635847131570823,
                                        4.32163613374937e-05, 0.0627274280370167, 66.4118798958804,
                                        122.361932206327, 0.372217324163812, 114.092521668926,
                                        308.106881011017, 63.6023224321684, 201.056970845445,
                                        201.27512854457, 98.9874539256948, 443.230119619488,
                                        20293.9092250464, 141.405866537237, 2487.84616355469])

        self.edinburghmedian = np.array([2.29180076e-04,   5.31804031e-01,   6.69448981e-02,
                                        4.46049258e-01,   1.18143120e+00,   5.31584216e-05,
                                        2.25487423e-03,   2.44782152e-03,   7.71092378e-05,
                                        3.82591095e-02,   7.47751776e+01,   1.16238252e+02,
                                        3.26252225e-01,   4.18554035e+01,   2.27257813e+02,
                                        1.20915004e+02,   1.15533213e+02,   1.27804720e+02,
                                        6.02259491e+01,   2.09997016e+02,   4.22672530e+03,
                                        3.67801053e+02,   1.62565304e+03])

        self.edinburghmean = np.array([ 9.80983217e-04,   5.19025559e-01,   1.08612889e-01,
                                        4.84356048e-01,   1.19950434e+00,   1.01336503e-04,
                                        3.22465935e-03,   3.44239452e-03,   1.11320287e-04,
                                        4.14726183e-02,   7.14355778e+01,   1.15778224e+02,
                                        3.20361827e-01,   4.13391057e+01,   2.20529309e+02,
                                        1.16768714e+02,   1.28460812e+02,   1.36541509e+02,
                                        6.86396830e+01,   2.83782534e+02,   6.50600814e+03,
                                        5.98832031e+02,   1.93625350e+03])

        self.ogpvals = np.array([4.41000000e-05,   4.70000000e-01,   2.80000000e-01,
                                1.60000000e-01,   1.50000000e+00,   3.60000000e-05,
                                2.48000000e-03,   2.28000000e-03,   2.60000000e-06,
                                6.93000000e-02,   1.50000000e+01,   4.04000000e+01,
                                5.06290000e-02,   3.00000000e+01,   1.97000000e+02,
                                9.00000000e+01,   5.20000000e+01,   4.18000000e+01,
                                5.80000000e+01,   1.02000000e+02,   7.70000000e+02,
                                4.00000000e+01,   9.89700000e+03])

        self.xb = self.edinburghmedian

        self.bnds = ((1e-5, 1e-2), (0.3, 0.7), (0.01, 0.5), (0.01, 0.5), (1.0001, 10.),
                     (2.5e-5, 1e-3), (1e-4, 1e-2), (1e-4, 1e-2), (1e-7, 1e-3), (0.018, 0.08),
                     (10., 100.), (60., 150.), (0.01, 0.5), (10., 100.), (242., 332.), (10., 100.), (10., 400.),
                     (10., 1000.), (10., 1000.), (10., 1000.), (100., 1e5), (10., 1000.), (100., 2e5))

        self.bnds2 = ((1e-5, 1e-2), (0.3, 0.7), (0.01, 0.5), (0.01, 0.5), (1.0001, 10.),
                     (2.5e-5, 1e-3), (1e-4, 1e-2), (1e-4, 1e-2), (1e-7, 1e-3), (0.018, 0.08),
                     (10, 100), (60, 150), (0.01, 0.5), (10, 100), (242, 332), (10, 150), (10, 400),
                     (10, 1000), (10, 1000), (10, 1000), (100, 1e5), (10, 1000), (100, 2e5))

        self.bnds3 = ((1e-5, 1e-2), (0.3, 0.7), (0.01, 0.5), (0.01, 0.5), (1.0001, 10.),
                     (2.5e-5, 1e-3), (1e-4, 1e-2), (1e-4, 1e-2), (1e-7, 1e-3), (0.018, 0.08),
                     (10, 100), (60, 150), (0.01, 0.5), (10, 100), (242, 332), (10, 150), (10, 400),
                     (10, 1000), (1, 1000), (10, 1000), (100, 1e5), (10, 1000), (100, 2e5))

        self.bnds4 = ((0, None), (0, None), (0, None), (0, None), (1.0001, 10.),
                     (0, None), (0, None), (0, None), (0, None), (0, None),
                     (0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None),
                     (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

        self.bnds5 = ((1e-5, 1e-2), (0.3, 0.7), (0.01, 0.5), (0.01, 0.5), (1.0001, 10.),
                     (2.5e-5, 1e-3), (1e-4, 1e-2), (1e-4, 1e-2), (1e-7, 1e-3), (0.018, 0.08),
                     (10, 100), (60, 150), (0.01, 0.5), (10, 100), (220, 332), (10, 150), (10, 400),
                     (30, 1000), (1, 1000), (10, 1000), (100, 1e5), (10, 1000), (100, 2e5))

        self.est_stdevs = np.array([(x[1]-x[0])/3 for x in self.bnds])

        self.est_xb = np.array([(x[1]+x[0])/2 for x in self.bnds])

        self.xa = None

        # Constants for ACM model
        self.acmwilliamsxls = np.array([0.0155, 1.526, 324.1, 0.2017,
                                        1.315, 2.595, 0.037, 0.2268,
                                        0.9576])
        self.acmreflex = np.array([0.0156935, 4.22273, 208.868, 0.0453194,
                                   0.37836, 7.19298, 0.011136, 2.1001,
                                   0.789798])
        self.a2 = None
        self.a3 = None
        self.a4 = None
        self.a5 = None
        self.a6 = None
        self.a7 = None
        self.a8 = None
        self.a9 = None
        self.a10 = None
        self.acm = self.acmreflex  # (currently using params from REFLEX)
        self.setacm(self.acm)
        self.phi_d = -2.5  # max. soil leaf water potential difference
        self.R_tot = 1.  # total plant-soil hydrolic resistance
        self.lat = 0.89133965  # latitutde of forest site in radians
        # lat = 51.153525 deg, lon = -0.858352 deg

        # 'Daily temperatures degC'
        self.t_mean = self.fluxdata['t_mean']
        self.t_max = self.fluxdata['t_max']
        self.t_min = self.fluxdata['t_min']
        self.t_range = np.array(self.t_max) - np.array(self.t_min)

        # 'Driving Data'
        self.I = self.fluxdata['i']  # incident radiation
        self.ca = 390.0  # atmospheric carbon
        self.D = self.fluxdata['day']  # day of year
        self.year = self.fluxdata['year']  # Year
        self.month = self.fluxdata['month']  # Month
        self.date = self.fluxdata['date']  # Date in month

        # misc
        self.radconv = 365.25 / np.pi

        # 'Background variances for carbon pools & B matrix'
        self.sigb_clab = 8.36**2  # (self.clab*0.2)**2 #20%
        self.sigb_cf = 11.6**2  # (self.cf*0.2)**2 #20%
        self.sigb_cw = 20.4**2  # (self.cw*0.2)**2 #20%
        self.sigb_cr = 154.**2  # (self.cr*0.2)**2 #20%
        self.sigb_cl = 8.**2  # (self.cl*0.2)**2 #20%
        self.sigb_cs = 1979.4**2  # (self.cs*0.2)**2 #20%
        self.backgstnddev = np.array([0.6, 0.5, 0.5, 0.5, 0.4, 0.6, 0.6, 0.6,
                                      0.6, 0.5, 0.5, 0.2, 0.5, 0.4, 0.1, 0.4,
                                      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        self.cpoolbstnddev = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1,
                                       0.1, 0.3, 0.3, 0.2, 0.1, 0.3, 0.1, 0.3,
                                       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        self.Bstdev = self.backgstnddev*self.xb

        self.x_truth_edc = np.array([2.66484233e-03,   6.89756644e-01,   1.66384448e-01,
                                     3.57888821e-01,   1.34684316e+00,   2.65047400e-04,
                                     3.67648598e-03,   4.18406120e-03,   3.50010595e-05,
                                     3.64738010e-02,   6.38965123e+01,   1.13600440e+02,
                                     3.52356927e-01,   3.06998625e+01,   2.62826792e+02,
                                     8.86500000e+01,   1.28740516e+02,   1.56612431e+02,
                                     1.23603777e+02,   1.59089365e+02,   3.66706797e+03,
                                     1.84384480e+02,   3.36554580e+03])

        self.x_pert_edc = np.array([2.41556892e-03,   5.80387867e-01,   2.04061310e-01,
                                    4.99646748e-01,   1.95792730e+00,   2.50002376e-05,
                                    5.51561999e-03,   7.43256295e-03,   1.40603907e-05,
                                    1.80045046e-02,   9.98910576e+01,   6.00020966e+01,
                                    1.00078125e-02,   3.83122903e+01,   2.42266816e+02,
                                    7.11298049e+01,   1.35739698e+02,   2.98781960e+02,
                                    6.61609244e+01,   5.52965103e+01,   3.07679533e+03,
                                    1.77940739e+02,   1.09547561e+03])

        self.edinburghstdev = np.array([2.03001590e-03,   1.16829160e-01,   1.11585876e-01,
                                        2.98860194e-01,   1.16141739e-01,   1.36472702e-04,
                                        2.92998472e-03,   3.11712858e-03,   1.18105073e-04,
                                        1.62308654e-02,   2.04219069e+01,   6.25696097e+00,
                                        1.14535431e-01,   1.40482247e+01,   3.72380005e+01,
                                        2.25938092e+01,   6.41030587e+01,   6.62621885e+01,
                                        3.59002726e+01,   2.19315727e+02,   7.14323513e+03,
                                        5.45013287e+02,   1.27646316e+03])

        #self.x_truth = np.array([1.00000000e-05,   3.13358350e-01,   3.00629189e-01,
        #                         4.45265166e-01,   1.02310470e+00,   1.22836138e-04,
        #                         5.04088931e-03,   1.56202990e-03,   1.48252124e-04,
        #                         7.61636968e-02,   9.27591545e+01,   1.22954168e+02,
        #                         1.00000000e-02,   4.67979617e+01,   2.87147216e+02,
        #                         5.51760150e+01,   5.16317404e+01,   1.00000000e+01,
        #                         1.00000000e+01,   5.01480167e+02,   7.26249788e+03,
        #                         6.26033838e+02,   2.35514838e+03])

        self.x_truth = np.array([1.28970000e-05,   3.13358350e-01,   3.00629189e-01,
                                 4.45265166e-01,   1.02310470e+00,   1.22836138e-04,
                                 5.04088931e-03,   1.56202990e-03,   1.48252124e-04,
                                 7.61636968e-02,   9.27591545e+01,   1.22954168e+02,
                                 1.28970000e-02,   4.67979617e+01,   2.87147216e+02,
                                 5.51760150e+01,   5.16317404e+01,   2.42000000e+01,
                                 8.90000000e+00,   5.01480167e+02,   7.26249788e+03,
                                 6.26033838e+02,   2.35514838e+03])

        self.x_truth2 = np.array([1.75177537e-03,   4.39216724e-01,   1.45275772e-01,
                                4.85994115e-01,   1.34919734e+00,   1.43577821e-04,
                                5.33065591e-03,   1.42217674e-03,   2.81251902e-04,
                                2.23716106e-02,   4.91216576e+01,   1.15606853e+02,
                                3.09940679e-01,   4.95177377e+01,   2.54598514e+02,
                                9.74817836e+01,   8.05741919e+01,   1.92147202e+02,
                                1.09382538e+02,   3.27096649e+02,   8.91617573e+03,
                                2.40016633e+02,   2.36359753e+03]) #from generated EDC ensem

        self.x_guess = np.array([2.87887370e-03,   5.27924849e-01,   2.01393985e-01,
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

        self.xb = self.edinburghmean

        #Model values
        self.pvallist = mod.mod_list(self.x_truth, self.d, 0, k*self.lenrun-1)

        self.test_stdev = np.array([0.5*(x[1]-x[0]) for x in self.bnds])
        self.test_xb = np.array([x[0]+0.5*(x[1]-x[0]) for x in self.bnds])

        self.B = self.makeb(self.test_stdev)
        # MAKE NEW B, THIS IS WRONG!

        #'Observartion variances for carbon pools and NEE'
        self.vars = np.array([self.clab*0.1, self.cf*0.1,
                              self.cw*0.1, self.cr*0.1,
                              self.cl*0.1, self.cs*0.1, 0.5, 0.2,
                              0.2, 0.2, 0.4, 0.12, 0.5, 0.5])
        self.smallvars = self.vars*1e-3

        if errs == 'normal':
            self.er = self.vars
        else:
            self.er = self.smallvars

        self.errdict = {'clab': self.er[0], 'cf': self.er[1],
                        'cw': self.er[2],'cl': self.er[3],'cr': self.er[4],
                        'cs': self.er[5], 'nee': self.er[6],
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
        if type(freq_list[0]) == int:
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
                    if self.erron == 1:
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

    def makeb(self, bstnddevs):
        """Creates diagonal B matrix.
        """
        bmat = (bstnddevs)**2*np.eye(23)
        return bmat
