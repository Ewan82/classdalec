import matplotlib.mlab as mlab
import numpy as np
import datetime
import xlrd
import csv
from numpy.lib.recfunctions import append_fields


def excel2csv(filename, yearst, yearend):
    """Given an excel spreadsheet with multiple years data in different work
    sheets, 'excel2csv' writes each years worth of data into a csv file.
    """
    for x in xrange(yearst, yearend+1):    
        with xlrd.open_workbook(filename) as wb:
            sh = wb.sheet_by_name(str(int(x)))  # or wb.sheet_by_name('name_of_the_sheet_here')
            with open('ahdat'+str(int(x))+'.csv', 'wb') as f:
                c = csv.writer(f)
                for r in range(sh.nrows):
                    c.writerow(sh.row_values(r))
        
    
def data(filename):
    """Extracts data from specified .csv file.
    """
    fluxdata = mlab.csv2rec(filename, missing = 'N/A') #create recorded array
    
    #Clip data    
    fluxdata['co2_flux\xb5molsm2'][fluxdata['co2_flux\xb5molsm2']>=50] = \
                                                                   float('NaN')
    fluxdata['co2_flux\xb5molsm2'][fluxdata['co2_flux\xb5molsm2']<=-50] = \
                                                                   float('NaN')
    fluxdata['rgwm2'][fluxdata['rgwm2']<0.] = 0.                                                            
   
    lenyear = len(fluxdata)/48
    
    serialdate=fluxdata['date_combined']
    seconds = (serialdate - 25569) * 86400.0 #converting from excel timestamp
    seconds2 = np.array([seconds[x*48] for x in xrange(lenyear)])
    datelist=np.array([datetime.datetime.utcfromtimestamp(abs(d)) for d in\
                       seconds2])

    year = np.ones(lenyear)*datelist[0].year
    day = np.arange(1, lenyear+1)
    month = np.ones(lenyear)*-9999
    date = np.ones(lenyear)*-9999 
    t_mean = np.ones(lenyear)*-9999
    t_max = np.ones(lenyear)*-9999
    t_min = np.ones(lenyear)*-9999
    I = np.ones(lenyear)*-9999
    nee = np.ones(lenyear)*-9999
    
    for x in xrange(0, lenyear):
        month[x] = datelist[x].month
        date[x] = datelist[x].day
        t_mean[x] = np.mean(fluxdata['tairdegc'][48*x:48*x+48])
        t_max[x] = np.max(fluxdata['tairdegc'][48*x:48*x+48])
        t_min[x] = np.min(fluxdata['tairdegc'][48*x:48*x+48])
        I[x] = 30*60*1e-6*np.sum(fluxdata['rgwm2'][48*x:48*x+48]) #Wm^-2 to MJm^-2day^-1
        fill = 0
        qcflag = 0
        if np.isnan(I[x])==True:
            I[x] = 0.9344*t_mean[x] + 1.0828
        for qc in xrange(48*x, 48*x+48):
            if np.isnan(fluxdata['co2_flux\xb5molsm2'][qc])==True:
                fill = fill + 1
            if fluxdata['qc_co2_flux'][qc] == 2:
                qcflag = qcflag + 1
        if fill > 0:
            nee[x] = float('NaN')
        elif qcflag > 1:
            nee[x] = float('NaN')
        else:
            nee[x] = (12.011)*1e-6*30*60*\
                           np.sum(fluxdata['co2_flux\xb5molsm2'][48*x:48*x+48])
            #micromol/s/m^2 to gC/m^2/day
    return np.array([year, month, date, day, t_mean, t_max, t_min, I, nee])


def soilresp(filename, ob):
    """Extracts data and converts to daily values for soilrootresp.
    """    
    soilroot = mlab.csv2rec(filename, missing='9999')
    lenarr=len(soilroot['year'])/24.
    year = np.ones(lenarr)*-9999
    day = np.ones(lenarr)*-9999
    month = np.ones(lenarr)*-9999
    date = np.ones(lenarr)*-9999
    soilresp = np.ones(lenarr)*-9999
    
    for x in xrange(0,int(lenarr)):
        year[x] = soilroot['year'][x*24]
        day[x] = soilroot['day_1'][x*24]
        month[x] = soilroot['month'][x*24]
        date[x] = soilroot['day'][x*24]
        fill = 0
        for qc in xrange(24*x, 24*x+24):
            if np.isnan(soilroot[ob][qc])==True:
                fill += 1
        if fill > 0:
            soilresp[x] = float('NaN')
        else:
            soilresp[x] = (12.011)*1e-6*60*60*\
                           np.sum(soilroot[ob][24*x:24*x+24])
    return np.array([year, month, date, day, soilresp]).T
    

def dat_output(filenames, outputname):
    """Extract data from multiple csv files and write to single csv file.
    """
    dat = data(filenames[0])
    for x in xrange(1,len(filenames)):
        dat = np.append(dat, data(filenames[x]), axis = 1)
    
    np.savetxt(outputname, dat.T, delimiter=',', fmt='%.4e', header="year," 
    "month, date, day, t_mean, t_max, t_min, I, nee", comments='')
    return dat
    
    
def add_obs(filename, obname, ob, obsfile, obsfunc, years):
    """Function adds more observations to current datafile.
    """
    dat = mlab.csv2rec(filename, missing='nan')
    dat2 = append_fields(dat, obname, np.ones(len(dat))*float('NaN'), 
                         fill_value=float('NaN'), usemask=False)
    newobdat=obsfunc(obsfile, ob) #create new ob array
    
    indexdata=[] #loop over dat2 to find indices for newobdat dates
    for x in xrange(len(dat2)):
        if dat2['year'][x]==newobdat[0,0] and dat2['month'][x]==newobdat[0,1] \
        and dat2['date'][x]==newobdat[0,2]:
            indexdata.append(x)
        elif dat2['year'][x]==newobdat[-1,0] and \
        dat2['month'][x]==newobdat[-1,1] and dat2['date'][x]==newobdat[-1,2]:
            indexdata.append(x)
    #takes fourth column of data as observation, make sure this is the case 
    #in obsfunc
    dat2[obname][indexdata[0]:indexdata[1]+1]=newobdat[:,4] 
    #write data to csv file
    mlab.rec2csv(dat2, filename, missing='nan')
    
    
                         
    
    
    
    