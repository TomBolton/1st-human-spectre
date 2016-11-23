##
##
##  Plot velocity power spectrum as a function of 
##  horizontal wavenumber
##
##  Borrows shamlessly from scripts by Rob Scott
##
##  Joakim Kjellsson, AOPP, May 2016
##
##

import os,sys,time
import numpy as np
from scipy.fftpack import fft, ifft, fftn, ifftn
from scipy.signal import periodogram, hamming, tukey
import scipy.stats as stats
from scipy.interpolate import griddata
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from psd_functions_new2 import *

def set_psd_fig(x='k'):
   fig = plt.figure()
   ax1  = fig.add_subplot(111)
   ax11 = ax1.twiny()
   
   if (x == 'k'):
      ax1.set_xlabel(r'Wavenumber $K$ $[\mathrm{m}^{-1}]$')
      ax11.set_xlabel(r'Wavelength $L = 2\pi/K$ $[\mathrm{km}]$')
   elif (x == 'o'):
      ax1.set_xlabel(r'Frequency $\omega$ $[\mathrm{days}^{-1}]$')
      ax11.set_xlabel(r'Time scale $[\mathrm{days}]$')
   
   return fig,ax1,ax11
   

def set_flux_fig():
   fig_1 = plt.figure(figsize=(8,10))
   ax_1  = fig_1.add_axes([0.2,0.7,0.4,0.18])
   ax_2  = fig_1.add_axes([0.2,0.5,0.4,0.18])
   ax_22 = fig_1.add_axes([0.2,0.3,0.4,0.18])
   ax_3  = fig_1.add_axes([0.2,0.1,0.4,0.18])
   ax_11 = ax_1.twiny()
   
   ax_1.spines['right'].set_visible(False)
   ax_1.spines['bottom'].set_visible(False)
   ax_1.yaxis.set_ticks_position('left')
   ax_1.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
   
   ax_11.spines['right'].set_visible(False)
   ax_11.spines['bottom'].set_visible(False)
   ax_11.xaxis.set_ticks_position('top')
   ax_11.tick_params(axis='x',bottom='off',labelbottom='off')
   
   ax_2.spines['right'].set_visible(False)
   ax_2.spines['top'].set_visible(False)
   ax_2.spines['bottom'].set_visible(False)
   ax_2.yaxis.set_ticks_position('left')
   ax_2.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
   #ax_22.spines['top'].set_visible(False)
   ax_22.spines['right'].set_visible(False)
   ax_22.yaxis.set_ticks_position('left')
   ax_22.spines['top'].set_visible(False)
   ax_22.spines['bottom'].set_visible(False)
   ax_22.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
   #ax_22.xaxis.set_ticks_position('bottom')
   
   ax_3.spines['top'].set_visible(False)
   ax_3.spines['right'].set_visible(False)
   ax_3.yaxis.set_ticks_position('left')
   ax_3.xaxis.set_ticks_position('bottom')
   
   ax_11.set_xlabel(r'Wavelength $2\pi/K$ $[\mathrm{km}]$')
   ax_3.set_xlabel(r'Wavenumber $K=\sqrt{k^2 + l^2}$ $[\mathrm{m}^{-1}]$')
   
   return fig_1,ax_1,ax_11,ax_2,ax_22,ax_3


def k2l(k):
    """
    Returns length given wavenumber k.
    """
    if (grid_type == 'xy'): km=1000
    elif(grid_type == 'll'): km=1000
    return (2.0 * np.pi / k)/km ## [km]

def k2t(k):
    """
    Returns time scale given wavenumber k.
    """
    return 2.0 * np.pi / k / 86400.


def convert_k_to_len(ax_f):
    """
    Update second axis according with first axis.
    """
    y1, y2 = ax1_ke.get_xlim()
    ax11_ke.set_xlim(k2l(y1), k2l(y2))
    ax11_ke.figure.canvas.draw()
    
    y1, y2 = ax1_psd.get_xlim()
    ax11_psd.set_xlim(k2l(y1), k2l(y2))
    ax11_psd.figure.canvas.draw()
    
    y1, y2 = ax1_psdt.get_xlim()
    ax11_psdt.set_xlim(k2t(y1), k2t(y2))
    ax11_psdt.figure.canvas.draw()
    
    y1, y2 = ax1_en.get_xlim()
    ax11_en.set_xlim(k2l(y1), k2l(y2))
    ax11_en.figure.canvas.draw()


def draw_lines(ax1,vk,psd,m0=False,m53=False,m3=False,m23=False):
   """
   """
   
   ## 
   ## Plot some example slopes
   ##
   if (jn == nt-1 and jn2 == nt2-1):
      if (m0):
         ## draw zero line
         ax1.semilogx(vk,vk*0,'-k')
            
      if (m53):
         ## draw -5/3 slope
         psdmax_m53 = vk.max()
         psdmin_m53 = psdmax_m53/50.
         kmax_m53 = vk[psd==psdmax_m53] * 1.5
         C_m53 = psdmax_m53/kmax_m53**(-5./3.)
         kmin_m53 = (psdmin_m53/C_m53)**(-3./5.)
         ax1.loglog([kmin_m53,kmax_m53],[psdmin_m53,psdmax_m53],\
                     '--k',linewidth=2.,label='k^(-5/3)')
            
      if (m3):      
         psdmax_m3 = psd.max()
         psdmin_m3 = psdmax_m3/50.
         kmax_m3 = vk[psd==psdmax_m3] * 2.
         C_m3 = psdmax_m3/kmax_m3**(-3.)
         kmin_m3 = (psdmin_m3/C_m3)**(-1./3.)
         ax1.loglog([kmin_m3,kmax_m3],[psdmin_m3,psdmax_m3],\
                    ':k',linewidth=2.,label='k^(-3)')
         

grid_type = 'll'
aver_type = 'baroclinic'
region    = 'kuroshio-full'
machine   = 'mac'
laviso    = False
lsmooth   = True
starttime = '20000105'
endtime   = '20000105'

noff = 0

if (machine == 'mac'):
   pdir = '/Users/joakim/Downloads/'
elif (machine == 'archer'):
   pdir = '/work/n01/n01/joakim2/frames/'
elif (machine == 'jasmin'):
   pdir = './'

klevels = np.arange(15,16)

m53 = False ## plot predicted -5/3 slope
m3  = True ## plot predicted -3 slope
m23 = False ## plot 2/3 slope
m0  = True  ## plot zero line for fluxes

lrhines = False    ## calculate and plot Rhines scale
lrossby = False    ## read and plot 1st baroclinic Rossby radius
lpsd_freq = False  ## store each step to make frequency spectrum analysis


## Diagnostics to plot
lke             = True  ## kinetic energy of top level
lke_freq        = False  ## frequency spectrum of kinetic energy of top level
lbke            = True  ## barotropic kinetic energy
lcke            = True  ## baroclinic kinetic energy
lTk_bc_bt       = True  ## spectral transfers of barotropic and baroclinic modes
lTk_wind_visc   = False  ## spectral wind forcing and dissipation

##
##
##

filenameU = []
filenameV = []
spectre_list = [] ## list with all analysed data

colors    = ['blue' ,'red'    ,'green'                   ,'cyan'    ]
style     = ['-',':','-.','--']
#name = ['ORCA1-N406','ORCA025-N401','ORCA0083-N001']
name = ['ORCA0083-N001']
#name = ['GYRE4_square','GYRE4_square_mem001']
orca = []

if (laviso): 
   name.append('AVISO')
   colors.append('magenta')

nt = len(name) ## number of datasets, not including -5/3 lines etc. 

if (m53): name.append('k^(-5/3)')
if (m3):  name.append('k^(-3)')
if (m23):  name.append('k^(2/3)')

cutoff_km = [60,15,15,5]

vyear, vmon, vday = find_time(starttime,endtime,5)
nt2 = vyear.shape[0]

print noff,nt2

for jn in range(0,nt):
   for jn2 in range(0,nt2):
      
      ##
      ## Data
      ##
      print jn,jn2
      
      if (1):   
         ##
         if (machine == 'jasmin'):
            ddir = '/group_workspaces/jasmin2/nemo/'
         elif (machine == 'archer'):
            ddir = '/work/n01/n01/joakim2/data/'
         elif (machine == 'mac'):
            ddir = '/Users/joakim/data/'
            
         #if (orca[jn] == '1'):
         if (name[jn] == 'ORCA1-N406'):
            prefixh = '/vol1/ORCA1-N406/domain/'
            prefixh2= '/ORCA1-N406/domain/'
            prefix1 = '/vol1/ORCA1-N406/means/%04d/' % (vyear[jn2],)
            prefix2 = '/ORCA1-N406/means/%04d/' % (vyear[jn2],)
            prefix = '/ORCA1-N406_%04d%02d%02dd05' % \
                     (vyear[jn2],vmon[jn2],vday[jn2])
            dlon = 1 ; dlat = 1
            scale = 1
            Ahm0 = 10000.
            
         #if (orca[jn] == '025'):
         if (name[jn] == 'ORCA025-N401'):
            prefixh = '/vol1/ORCA025-N401/domain/'
            prefixh2 = '/ORCA025-N401/domain/'
            prefix1 = '/vol1/ORCA025-N401/means/%04d/' % (vyear[jn2],)
            prefix2 = '/ORCA025-N401/means/%04d/' % (vyear[jn2],)
            prefix = '/ORCA025-N401_%04d%02d%02dd05' % \
                     (vyear[jn2],vmon[jn2],vday[jn2])
            dlon = 0.25 ; dlat = 0.25
            scale = 4
            Ahm0 = -2.2 * 10**11
                  
         #if (orca[jn] == '025-cg'):
         if (name[jn] == 'ORCA0083-CG'):
            prefixh = '/vol1/ORCA025-N401/domain/'
            prefixh2 = '/ORCA025-N401/domain/'
            prefix1 = '/vol4/joakim/coarse_grain/'
            prefix2 = '/ORCA0083-N001/'
            prefix = '/ORCA0083-N01_%04d%02d%02dd05' % \
                     (vyear[jn2],vmon[jn2],vday[jn2])
            dlon = 0.25 ; dlat = 0.25
            scale = 4
            Ahm0 = -2.2 * 10**11
            
         #if (orca[jn] == '0083'):
         if (name[jn] == 'ORCA0083-N001'):
            prefixh = '/vol1/ORCA0083-N001/domain/'
            prefixh2 = '/ORCA0083-N001/domain/'
            prefix1 = '/vol1/ORCA0083-N001/means/%04d/' % (vyear[jn2],)
            prefix2 = '/ORCA0083-N001/means/%04d/' % (vyear[jn2],)
            prefix = '/ORCA0083-N01_%04d%02d%02dd05' % \
                     (vyear[jn2],vmon[jn2],vday[jn2])   
            dlon = 1/12. ; dlat = 1/12.
            scale = 12
            Ahm0 = -1.25 * 10**10
            
         if (name[jn] == 'AVISO'):
            dlon = 1/4. ; dlat = 1/4.
            Ahm0 = 0.
         
         if (name[jn] == 'GYRE4_square' or \
            name[jn] == 'GYRE4_square_mem001'):
            prefixh = '/GYRE4_square/'
            prefix1 = '/GYRE4_square/SAVED/'
            prefix2 = '/GYRE4_square/SAVED/'
            prefix = '/'+name[jn]+'/'+name[jn]+'_5d_00090101_00091230_grid_'
            dlon = 1/4. ; dlat = 1/4.
            scale = 4
            Ahm0 = -2.2 * 10**11
             
         if (name[jn] == 'GYRE12_square' or \
             name[jn] == 'GYRE12_square_mem-0.01_bilap' or \
             name[jn] == 'GYRE12_square_mem-0.005_bilap' or \
             name[jn] == 'GYRE12_square_mem-0.03_bilap' or \
             name[jn] == 'GYRE12_square_mem_gamma-2_kappa0_bilap' or \
             name[jn] == 'GYRE12_square_mem_gamma-1_kappa1_bilap' or \
             name[jn] == 'GYRE12_square_mem001' or \
             name[jn] == 'GYRE12_square_mem003'):
             prefixh = '/GYRE12_square/'
             prefix1 = '/GYRE12_square/SAVED/'
             prefix2 = '/GYRE12_square/SAVED/'
             prefix = '/'+name[jn]+'/'+name[jn]+'_5d_00090101_00091230_grid_'
             dlon = 1/12. ; dlat = 1/12.
             scale = 12
             Ahm0 = -1.25 * 10**10
            
         #if (grid_type == 'll'):
         #if (orca[jn] == '1'):
         #   dlon = 1 ; dlat = 1
         #   scale = 1
         #elif (orca[jn][0:3] == '025'):
         #   dlon = 0.25 ; dlat = 0.25
         #   scale = 4
         #elif (orca[jn] == '0083'):
         #   dlon = 1/12. ; dlat = 1/12.
         #   scale = 12
         #elif (orca[jn] == 'AVISO'):
         #   dlon = 0.25 ; dlat = 0.25
         #else:
         #   print ' unknown resolution '
         #   sys.exit()
         
      
         
         if (name[jn] != 'AVISO'):
            if (machine == 'jasmin'): 
               prefix = prefix1+prefix
               hfile  = ddir + prefixh 
               zfile  = ddir + prefixh 
            elif (machine == 'mac' or machine == 'archer'):
               prefix = prefix2 + prefix
               hfile  = ddir + prefixh2
               zfile  = ddir + prefixh2
            
            hfile = hfile + 'mesh_hgr.nc'
            zfile = zfile + 'mesh_zgr.nc'
            
         if (name[jn] == 'ORCA0083025-CG'):
            ufile = ddir + prefix + 'U_coarse.nc'
            vfile = ddir + prefix + 'V_coarse.nc'
         elif (name[jn] != 'AVISO'):
            ufile = ddir + prefix + 'U.nc'
            vfile = ddir + prefix + 'V.nc'
      
      
      if (jn2 == 0):
         lon0,lon1,lat0,lat1 = set_region(region)
         print lon0,lon1,lat0,lat1,region
         if (name[jn] != 'AVISO'):
            ulon,ulat,vlon,vlat,tlon,tlat = read_nemo_grid(hfile,0,-1,0,-1)
            if (lon1 < lon0): 
               tmp_lon1 = lon1+360
               tlon = np.where(tlon<0,tlon+360,tlon)
            else:
               tmp_lon1 = lon1
            ind = np.where( (tlon>=lon0) & (tlon<=tmp_lon1) & (tlat>=lat0) & (tlat<=lat1) )
            if (lon1 < lon0): 
               tlon = np.where(tlon>180,tlon-360,tlon)
            imt = tlon.shape[1]
            jmt = tlon.shape[0]
            i_ind = ind[1]
            j_ind = ind[0]
            i0 = i_ind.min()
            i1 = i_ind.max()
            j0 = j_ind.min()
            j1 = j_ind.max()
            
         elif (name[jn] == 'AVISO'):
            if (lon0 < 0):
               tmp_lon0 = lon0+360
            else:
               tmp_lon0 = lon0
            if (lon1 < 0):
               tmp_lon1 = lon1+360
            else:
               tmp_lon1 = lon1
            i0 = int( (tmp_lon0-0.125)/0.25 )
            i1 = int( (tmp_lon1-0.125)/0.25 )
            j0 = int( (lat0+89.875)/0.25 )
            j1 = int( (lat1+89.875)/0.25 )
         
         if (i1 < i0):
            i_min = [i0,0]
            i_max = [imt,i1]
            j_min = [j0,j0]
            j_max = [j1,j1]
         else:
            i_min = [i0]
            i_max = [i1]
            j_min = [j0]
            j_max = [j1]
      
      #if (lon1 > 180):
      #   lon1 = lon1-360
      #if (lon0 > 180):
      #   lon0 = lon0-360   
      #i0,i1,j0,j1 = set_ij(region,orca[jn])
      
      print name[jn]
      for ji in range(0,len(i_min)):
         if (name[jn] == 'AVISO'):
            i0 = i_min[ji]
            j0 = j_min[ji]
            i1 = i_max[ji]
            j1 = j_max[ji]
            
            print i0,i1,j0,j1
            print i0*0.25+0.125,i1*0.25+0.125,j0*0.25-89.875,j1*0.25-89.875
            time1 = '%04d%02d%02d' % (vyear[jn2],vmon[jn2],vday[jn2])
            time2 = '%04d%02d%02d' % (vyear[jn2],vmon[jn2],vday[jn2])
            ulon,ulat,uvel_full,vvel_full = read_aviso(time1,time2,i0,i1,j0,j1)
            vlon = ulon.copy()
            vlat = ulat.copy()
            
         else:
            i0 = i_min[ji]
            i1 = i_max[ji]
            j0 = j_min[ji]
            j1 = j_max[ji]
            print i0,i1,j0,j1
            print tlon[j0,i0],tlon[j0,i1],tlat[j0,i0],tlat[j1,i0]
            if (jn2 == 0): dzt,dz,mbathy = read_nemo_zgrid(zfile,i0,i1,j0,j1)
            ulon,ulat,vlon,vlat,uvel_full,vvel_full,taux_full,tauy_full = read_nemo(ufile,vfile,i0,i1,j0,j1,\
                                                                          aver_type=aver_type) 
            
            if (lrossby and jn2 == 0 and name[jn] == 'ORCA1-N406' and machine == 'jasmin'):
               nc = Dataset('/group_workspaces/jasmin2/aopp/joakim/rossby_radius/rossby_radius_20000105-20001231.nc','r')
               rad = nc.variables['rossby_radius'][:,j0:j1,i0:i1]
               Ld = np.mean(rad)
               dLd = np.std(rad)  
               print ' 1st Rossby radius, ',Ld 
            
      if (machine == 'archer'):
         cmap = plt.cm.YlGnBu_r
      elif (machine == 'jasmin'):
         cmap = plt.cm.YlGnBu_r
      elif (machine == 'mac'):
         cmap = plt.cm.viridis
            
      ## Plot map of KE and mark region of interest
      if (jn2 == 0):
         fig = plt.figure()
         ax  = fig.add_subplot(111)
         ax.set_title('Surface kinetic energy in '+name[jn])
         levs = np.arange(0,0.2,0.02)
         ke = 0.5 * ( uvel_full[0,0,:,:]**2 + vvel_full[0,0,:,:]**2   )            
         ke = 0.5 * ( taux_full[0,:,:]**2 + tauy_full[0,:,:]**2   )
         cf  = ax.contourf(ke,extend='max',cmap=cmap,levels=levs)
         ax.plot([i0,i1,i1,i0,i0],[j0,j0,j1,j1,j0],'-r')
            
         xp,yp = rotate_box([i0,i1,i1,i0],[j0,j0,j1,j1],45)
         xp.append(xp[0])
         yp.append(yp[0])
         #ax.plot(xp,yp,'-r')
               
         plt.colorbar(cf,ax=ax)
         fig.savefig(pdir+'map_'+region+'_'+name[jn]+'.png',format='png')
      
                                                         
      if (1):      
         xmin = lon0#np.max(ulon[:,0])
         xmax = lon1#np.min(ulon[:,-1])
         ymin = lat0#np.max(ulat[0,:])
         ymax = lat1#np.min(ulat[-1,:])
         
         print lon0,lon1
         #print xmin, xmax, ymin, ymax, region
         if (lon1 < lon0): 
            #lon = np.where( ulon < 0, ulon+360, ulon )
            xmin = lon0 #np.max(lon[:,0])
            xmax = lon1 + 360 #np.min(lon[:,-1])
            #ymin = np.max(ulat[0,:])
            #ymax = np.min(ulat[-1,:])
            #print xmin, xmax, ymin, ymax
            
         xx = np.arange(xmin,xmax+dlon,dlon)
         yy = np.arange(ymin,ymax+dlat,dlat)
         
         print xx[0],xx[-1],dlon
         print yy[0],yy[-1],dlat
         
      
      ## Set wavenumbers in lon-lat
      xx2,yy2,wn_x,wn_y,kx,ky,k,dk = set_wavenumbers_ll(lon0,lon1,lat0,lat1,dlon=dlon,dlat=dlat)
      nx = xx2.shape[1]
      ny = xx2.shape[0]
      ## Set wavenumbers in x-y
      ## Assumes dx constant with latitude, i.e. dx = dlon * radius * cos(mean_lat)
      ## This seems ok as long as the region of interest does not cover more than ca. 20 degrees
      tmp1,tmp2,wn_x,wn_y,kx,ky,k,dk = set_wavenumbers_xy(lon0,lon1,lat0,lat1,dlon=dlon,dlat=dlat)
      nk = k.shape[0]
      wvsq  = kx**2 + ky**2
      
      print 'Length 0 and -1', 2*np.pi/k[0],2*np.pi/k[-1]
      
      if (name[jn][0:13] == 'GYRE12_square'):
         uvel = np.ma.masked_array(uvel_full[jn2+noff,:,:,:].copy())
         vvel = np.ma.masked_array(vvel_full[jn2+noff,:,:,:].copy())
         taux = np.ma.masked_array(taux_full[jn2+noff,:,:,:].copy())
         tauy = np.ma.masked_array(tauy_full[jn2+noff,:,:,:].copy())
      else:
         uvel = np.ma.masked_array(uvel_full[0,:,:,:].copy())
         vvel = np.ma.masked_array(vvel_full[0,:,:,:].copy())
         taux = np.ma.masked_array(taux_full[0,:,:].copy())
         tauy = np.ma.masked_array(tauy_full[0,:,:].copy())
      
      print uvel.shape,vvel.shape,xx2.shape,yy2.shape,nx,ny
      
      ##
      ## Normalise weights for u,v
      ##
      #fig = plt.figure()
      #ax1 = fig.add_subplot(111)
      #ax1.plot(uvel[:,9,30],np.arange(0,75),label='before av')
      dztmean = np.mean(dzt,axis=0)
      for jk in range(0,uvel.shape[0]):
         uvel[jk,:,:] = uvel[jk,:,:] * dzt[jk,:,:] / dztmean[:,:]
         vvel[jk,:,:] = vvel[jk,:,:] * dzt[jk,:,:] / dztmean[:,:]
      #ax1.plot(uvel[:,9,30],np.arange(0,75),label='after av')   
      #ax1.legend()
      #plt.show()
      
      if (lrhines):
         if (jn == 0):
            Lr = []
         u = np.ma.sqrt( np.ma.mean( uvel**2 + vvel**2 ) )
         b = 2. * 7.2921150 * 1e-5 * np.cos( np.pi/180. * mean_lat ) / (6371.0*1000)
         r = 2 * np.pi * np.sqrt( u / b )
         if (jn2 == 0):
            vr = np.array([])
         vr = np.append(vr,r)
         print 'Rhines scale ',r,vr.shape
         if (jn2 == nt2-1):
            Lr.append(vr.mean())
            print 'Mean Rhines scale ',Lr,u
         
      ## If grid is lon-lat, 
      ## we interpolate to regular lon-lat grid
      ## and then use the wavenumbers in x-y from above
      if (grid_type == 'll'):# and orca[jn] != 'AVISO'):
         uvel_xy = np.zeros((uvel.shape[0],xx2.shape[0],xx2.shape[1]))
         vvel_xy = np.zeros((uvel.shape[0],xx2.shape[0],xx2.shape[1]))
         for jk in range(0,uvel.shape[0]):
            t0 = time.time()
            lst = [ulon,ulat,uvel[jk,:,:],xx2,yy2]
            uvel_xy[jk,:,:] = interp(lst)
            lst = [vlon,vlat,vvel[jk,:,:],xx2,yy2]
            vvel_xy[jk,:,:] = interp(lst)
            t1 = time.time()
            print ' Interpolated to regular grid in ',t1-t0,' s'
            
         lst = [ulon,ulat,taux,xx2,yy2]
         taux = interp(lst)
         lst = [vlon,vlat,tauy,xx2,yy2]
         tauy = interp(lst)
         
      uvel = np.ma.masked_array(uvel_xy.copy())
      vvel = np.ma.masked_array(vvel_xy.copy())
         
      uvel[uvel.mask] = 0.
      vvel[vvel.mask] = 0.
      taux[taux.mask] = 0.
      tauy[tauy.mask] = 0.
      
      ## Save uvel, vvel, taux, tauy
      if (lpsd_freq):
         if (jn2 == 0):
            uvel_store = uvel[np.newaxis,:,:,:]
            vvel_store = vvel[np.newaxis,:,:,:]
            taux_store = taux[np.newaxis,:,:,:]
            tauy_store = tauy[np.newaxis,:,:,:]
         else:
            uvel_store = np.concatenate( (uvel[np.newaxis,:,:,:],uvel_store), axis=0 )
            vvel_store = np.concatenate( (vvel[np.newaxis,:,:,:],vvel_store), axis=0 )
            taux_store = np.concatenate( (taux[np.newaxis,:,:,:],taux_store), axis=0 )
            tauy_store = np.concatenate( (tauy[np.newaxis,:,:,:],tauy_store), axis=0 )
      
      ## Calculate relative vorticity
      vort = np.zeros((uvel.shape[0],uvel.shape[1],uvel.shape[2]))
      for jk in range(0,uvel.shape[0]):
         vort[jk,:,:] = calculate_vorticity(uvel[jk,:,:],vvel[jk,:,:],xx2,yy2)
      
      if (jn2 == 0):
         fig = plt.figure()
         ax = fig.add_subplot(1,1,1)
         ax.set_title(name[jn])
         cf = ax.contourf(vort[0,:,:],levels=np.arange(-0.2,0.22,0.02),cmap=cmap,extend='both')
         plt.colorbar(cf,ax=ax)
      
      ## Window function
      #window_x = hamming(nx)
      #window_y = hamming(ny)
      window_x = tukey(nx)
      window_y = tukey(ny)
      window_2D = np.meshgrid(window_x,window_y)[0]
      window_fft = fftn(window_2D)
      
      ## Fourier transform of velocities
      #uhat = fftn(uvel*window_2D**2)
      #vhat = fftn(vvel*window_2D**2)
      #uhat = fftn(uvel) * window_fft
      #vhat = fftn(vvel) * window_fft
      
      ##
      ## Calculations 
      ##
      
      print ' Calculate zonal and vertical means and anomalies'
      means_anomalies = calculate_means_and_anomalies(uvel,vvel,dzt)
         
      ## Barotropic KE (includes zonal mean)
      bke_2D = calculate_ke(kx,ky,means_anomalies['u_vm'],means_anomalies['v_vm'])
      ## Barotropic EKE (deviations from zonal mean)
      beke_2D = calculate_ke(kx,ky,means_anomalies['u_za_vm'],means_anomalies['v_za_vm'])
      ## Baroclinic KE and EKE 
      cke_2D = np.zeros((beke_2D.shape))
      ceke_2D = np.zeros((beke_2D.shape))
      for jk in range(0,uvel.shape[0]):
         cke_tmp = calculate_ke(kx,ky,means_anomalies['u_va'][jk,:,:],\
                                      means_anomalies['v_va'][jk,:,:]) / float(uvel.shape[0])
         ceke_tmp = calculate_ke(kx,ky,means_anomalies['u_za_va'][jk,:,:],\
                                       means_anomalies['v_za_va'][jk,:,:]) / float(uvel.shape[0])
         cke_2D  = cke_2D  + cke_tmp
         ceke_2D = ceke_2D + ceke_tmp
         
      psd_2D = calculate_ke(kx,ky,means_anomalies['u_vm'],means_anomalies['v_vm'])
      
      ## Calculate spectral fluxes of triad interactions
      ## Change in barotropic EKE from baroclinic interactions
      Tk_data = calculate_spectral_flux_baroclinic_barotropic(kx,ky,means_anomalies['u_za_vm'],means_anomalies['v_za_vm'],\
                                                                    means_anomalies['u_za_va'],means_anomalies['v_za_va'])
      
      adv_bt_bc_bc = Tk_data['Tk_bt_bc_bc']
      adv_bt_bt_bt = Tk_data['Tk_bt_bt_bt']
      adv_bc_bc_bc = Tk_data['Tk_bc_bc_bc']
      
      uvel = uvel[0,:,:]
      vvel = vvel[0,:,:]
      vort = vort[0,:,:]
      
      ke_adv = calculate_spectral_flux(kx,ky,uvel,vvel)
      ens_2D = calculate_ens(kx,ky,vort)
      ens_adv = calculate_spectral_ens_flux(kx,ky,uvel,vvel,vort)
      
      Ahm = np.ones(uvel.shape) * Ahm0
      if (dlon >= 0.5): order='2'
      else: order='4'
      ke_visc_u = calculate_spectral_viscosity(kx,ky,uvel,Ahm,order=order)
      ke_visc_v = calculate_spectral_viscosity(kx,ky,vvel,Ahm,order=order)
      ke_visc   = ke_visc_u + ke_visc_v
      
      ## Forcing
      ke_tau = calculate_spectral_forcing(kx,ky,uvel,vvel,taux,tauy,rho=1023)
      
      ## Divide by grid box size
      #psd_2D = psd_2D / (nx**2 * ny**2)
      ke_adv = ke_adv / (nx**2 * ny**2)
      #ens_2D = ens_2D / (nx**2 * ny**2)
      ens_adv = ens_adv / (nx**2 * ny**2)
      ke_visc = ke_visc / (nx**2 * ny**2)
      ke_tau  = ke_tau / (nx**2 * ny**2)
      
      ##
      ## Integrate 2D PSD around lines of constant k
      ## 
      ## For spectral KE flux, we here define PI(k)
      ## which is the transfer of energy from all 
      ## scales of wavenumber < k (large scales) to 
      ## those of wavenumber > k (small scales)
      ## PI = \int_k^inf T(k) dk or
      ## PI = -\int_0^k T(k) dk
      ## but since we don't have the largest scales (smallest k)
      ## we need to integrate from infinity to k
      ## where T(k) is e.g. -u_conj*u*du/dx 
      ## and dE/dt = u_conj*du/dt
      ## It follows that dE/dt = -dPI/dk
      ##
      
      #
      # psd_ke [m3/s2]
      # Tk_ke  [m3/s3]
      # Tk_tau [m3/s3]
      # Tk_visc[m3/s3]
      #
      # Spectral energy densiy budget
      # d/dt psd_ke = Tk_ke + Tk_tau + Tk_visc
      #
      # Pi_ke  [m2/s3]
      #
      # Spectral energy budget
      # d/dt KE = Pi_ke + Pi_tau + Pi_visc
      #
      vk,psd_ke,ke_1D   = integrate_spectrum(psd_2D,wvsq,k,dk)
      vk,psd_ens,ens_1D = integrate_spectrum(ens_2D,wvsq,k,dk)
      vk,Tk_ke,Pi_ke    = integrate_spectrum(ke_adv,wvsq,k,dk)
      vk,Tk_ens,Pi_ens  = integrate_spectrum(ens_adv,wvsq,k,dk)
      vk,Tk_visc,Pi_visc = integrate_spectrum(ke_visc,wvsq,k,dk)
      vk,Tk_tau,Pi_tau  = integrate_spectrum(ke_tau,wvsq,k,dk)
      
      vk,psd_bke ,bke  = integrate_spectrum(bke_2D ,wvsq,k,dk)
      vk,psd_beke,beke = integrate_spectrum(beke_2D,wvsq,k,dk)
      vk,psd_cke ,cke  = integrate_spectrum(cke_2D ,wvsq,k,dk)
      vk,psd_ceke,ceke = integrate_spectrum(ceke_2D,wvsq,k,dk)
      
      vk,Tk_bt_bc_bc,Pi_bt_bc_bc = integrate_spectrum(adv_bt_bc_bc,wvsq,k,dk)
      vk,Tk_bt_bt_bt,Pi_bt_bt_bt = integrate_spectrum(adv_bt_bt_bt,wvsq,k,dk)
      vk,Tk_bc_bc_bc,Pi_bc_bc_bc = integrate_spectrum(adv_bc_bc_bc,wvsq,k,dk)
      
      ens = np.zeros((nk))
      ens2 = np.zeros((nk))
      ## Calculate PI(k) 
      for jk in range(0,nk):
         indices = np.where(wvsq >= k[jk]**2) 
         ens[jk] = np.sum(ens_2D[indices]) 
         ens2[jk] = np.sum(ens_adv[indices])
         
      ens1D  = -(ens[1:] - ens[0:-1]) / dk
      ens_adv1D = -(ens2[1:]-ens2[0:-1]) / dk
      ens_adv1D2 = 0.5*(ens2[1:]+ens2[0:-1])
      
      ## Calculate time scale, 
      tt_ke = np.zeros((nk-1))
      for jk in range(0,nk-1):
         tt_ke[jk] = 1./(np.sqrt(psd_ke[jk]*dk)*vk[jk]) / 86400. 
      
      ##
      ## Plot power spectrum
      ##
      
      ##
      ## Store means
      ##
      if (jn2 == 0):
         print ' Set averages to zero '
         ## Array for storing
         i_psd_bke = 10
         i_psd_beke = 11
         i_psd_cke = 12
         i_psd_ceke = 13
         i_Tk_bt_bc_bc = 14
         i_Tk_bt_bt_bt = 15
         i_Tk_bc_bc_bc = 16
         
         store = np.zeros((20,nt,nt2,nk-1))
         
      
      print ' Store data '    
      store[0,jn,jn2,:] = psd_ke[:]
      store[1,jn,jn2,:] = Tk_ke[:]
      store[2,jn,jn2,:] = ens1D[:]
      store[3,jn,jn2,:] = ens_adv1D[:]
      store[4,jn,jn2,:] = Pi_ke[:]
      store[5,jn,jn2,:] = ens_adv1D2[:]
      store[6,jn,jn2,:] = Tk_visc[:]
      store[7,jn,jn2,:] = Pi_visc[:]
      store[8,jn,jn2,:] = Tk_tau[:]
      store[11,jn,jn2,:] = tt_ke[:]
      store[i_psd_bke ,jn,jn2,:] = psd_bke[:]
      store[i_psd_beke,jn,jn2,:] = psd_beke[:]
      store[i_psd_cke ,jn,jn2,:] = psd_cke[:]
      store[i_psd_ceke,jn,jn2,:] = psd_ceke[:]
      store[i_Tk_bt_bc_bc,jn,jn2,:] = Tk_bt_bc_bc[:]
      store[i_Tk_bt_bt_bt,jn,jn2,:] = Tk_bt_bt_bt[:]
      store[i_Tk_bc_bc_bc,jn,jn2,:] = Tk_bc_bc_bc[:]
      
      ## smooth using a 20-point running mean, or less if
      ## data is less than 20 points. 
      if (lsmooth):
         for jk in range(0,store.shape[0]):
            data_list = [store[jk,jn,jn2,:]]
            [store[jk,jn,jn2,:]] = smooth_spectrum(data_list,n=20)
      
      if (jn2 == nt2-1):
         ## Calculate average
         #psd_ke  = np.mean(store[0,jn,:,:],axis=0)
         #Tk_ke   = np.mean(store[1,jn,:,:],axis=0)
         #Pi_ke   = np.mean(store[4,jn,:,:],axis=0)
         #Tk_tau  = np.mean(store[8,jn,:,:],axis=0)
         #Tk_visc = np.mean(store[6,jn,:,:],axis=0)
         #Pi_visc = np.mean(store[7,jn,:,:],axis=0)
         
         mean_store = np.mean(store,axis=2)
         psd_ke  = mean_store[0,jn,:]
         Tk_ke   = mean_store[1,jn,:]
         Pi_ke   = mean_store[4,jn,:]
         Tk_tau  = mean_store[8,jn,:]
         Tk_visc = mean_store[6,jn,:]
         Pi_visc = mean_store[7,jn,:]
         tt_ke   = np.mean(store[11,jn,:,:],axis=0)
         
         psd_bke  = mean_store[i_psd_bke ,jn,:]
         psd_beke = mean_store[i_psd_beke,jn,:]
         psd_cke  = mean_store[i_psd_cke ,jn,:]
         psd_ceke = mean_store[i_psd_ceke,jn,:]
         
         Tk_bt_bc_bc = mean_store[i_Tk_bt_bc_bc,jn,:]
         Tk_bt_bt_bt = mean_store[i_Tk_bt_bt_bt,jn,:]
         Tk_bc_bc_bc = mean_store[i_Tk_bc_bc_bc,jn,:]
         
         ## Calculate variability
         psd_ke_std = np.std(store[0,jn,:,:],axis=0)
         Tk_ke_std  = np.std(store[1,jn,:,:],axis=0)
         Tk_tau_std = np.std(store[8,jn,:,:],axis=0)
         Tk_visc_std= np.std(store[6,jn,:,:],axis=0)
         Pi_ke_std  = np.std(store[4,jn,:,:],axis=0)
         
         vens    = np.mean(store[2,jn,:,:],axis=0)
         vens_adv = np.mean(store[3,jn,:,:],axis=0)
         vens_adv2 = np.mean(store[5,jn,:,:],axis=0)
         
         p1 = 10 ; p2 = 90
         psd_ke_p1  = np.percentile(store[0,jn,:,:],p1,axis=0)
         psd_ke_p2  = np.percentile(store[0,jn,:,:],p2,axis=0)
         Tk_ke_p1   = np.percentile(store[1,jn,:,:],p1,axis=0)
         Tk_ke_p2   = np.percentile(store[1,jn,:,:],p2,axis=0)
         Tk_tau_p1  = np.percentile(store[8,jn,:,:],p1,axis=0)
         Tk_tau_p2  = np.percentile(store[8,jn,:,:],p2,axis=0)
         Tk_visc_p1 = np.percentile(store[6,jn,:,:],p1,axis=0)
         Tk_visc_p2 = np.percentile(store[6,jn,:,:],p2,axis=0)
         Pi_ke_p1   = np.percentile(store[4,jn,:,:],p1,axis=0)
         Pi_ke_p2   = np.percentile(store[4,jn,:,:],p2,axis=0)
         
         print 'vadv',Tk_ke
         
         ## Calculate frequency spectra
         if (lpsd_freq):
            Lt = nt2 * 5. * 86400. 
            if (np.mod(nt2,2) == 0):
               arr = np.concatenate( (np.arange(0,nt2/2+1),np.arange(-nt2/2+1,0)) )
            else:
               nnt = (nt2-1)/2
               arr = np.concatenate( (np.arange(0,nnt+1),np.arange(-nnt,0)) )
            wn_t = 2.0 * np.pi / Lt * arr
            wn_max = np.max( np.sqrt(wn_t**2) )
            dk = 2.0*np.pi/Lt
            k_t = np.arange(dk,wn_max+dk,dk)
            
            print uvel_store.shape
            uhat = fft(uvel_store,axis=0)
            vhat = fft(vvel_store,axis=0)
            print uhat.shape
            ke_t = np.real(np.conj(uhat) * uhat) + np.real(np.conj(vhat) * vhat)
            
            nk_t = k_t.shape[0]
            psc = np.zeros((nk_t))
            ke1D_t = np.zeros((nk_t))
            for jk in range(0,nk_t):
               steps = np.where(wn_t**2 >= k_t[jk]**2)[0]
               psc[jk] = np.sum(ke_t[steps,:,:])
               steps = np.where(wn_t**2 == k_t[jk]**2)[0] 
               ke1D_t[jk] = np.sum( ke_t[steps,:,:] ) / (nx*ny)
               print steps
               
            psc = psc / (nx*ny)
            print ke_t.shape,psc.shape
            psd_t = -(psc[1:] - psc[0:-1]) / dk
            psd_t[psd_t == 0] = 0.
            print 'psd_t min, max ',psd_t.min(),psd_t.max()
            print 'psd_t <= 0',psd_t[psd_t<=0]
            #ke1D_t = np.mean( np.mean(ke_t,axis=1), axis=1 )
            print 'ke_t <= 0',ke1D_t[ke1D_t<=0]
            for jk in range(0,nk_t-1):
               if (psc[jk] == psc[jk+1]):
                  print jk,psc[jk],ke1D_t[jk],ke1D_t[jk+1],psd_t[jk]
                  
            ke_t  =  (psc[1:] + psc[0:-1]) * 0.5 
            ke1D_t  =  (ke1D_t[1:] + ke1D_t[0:-1]) * 0.5 
            vk_t = 0.5 * (k_t[1:] + k_t[0:-1])
            print ke1D_t.shape
            print wn_t.shape
            
            #ax1_psdt.loglog(wn_t,ke1D_t/dk,label=name[jn],color=colors[jn],linestyle='-',basex=10,basey=10)
            #ax1_psdt.loglog(vk_t,psd_t,label='psd',basex=10,basey=10,color='k')
            #ax1_psdt.loglog(vk_t,ke_t,label='ke_t',basex=10,basey=10,color='r')
            ax1_psdt.semilogy(vk_t,ke1D_t/dk,label=name[jn],color=colors[jn],linestyle='-')
            ax1_psdt.semilogy(vk_t,psd_t,label='psd',color='k')
            ax1_psdt.semilogy(vk_t,ke_t,label='ke_t',color='r')
            ax1_psdt.legend()
            
            y0,y1 = ax1_psdt.get_ylim()
            #ax1_psdt.loglog([2*np.pi/365.,2*np.pi/365.],[y0,y1],'--k')
            
            ##
            ## Also, plot surface plot of KE and std
            ##
            
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)
            ax1.set_title('Mean KE')
            ax2.set_title('Std KE')
            cf1 = ax1.pcolormesh(np.mean(uvel_store**2 + vvel_store**2,axis=0),vmin=-0.1,vmax=0.1)
            plt.colorbar(cf1,ax=ax1)
            cf2 = ax2.pcolormesh(np.std( uvel_store**2 + vvel_store**2,axis=0),vmin=0,vmax=0.1)
            plt.colorbar(cf2,ax=ax2)
         
         
         steps = np.arange(0,psd_ke.shape[0])
         
         
         ##
         ## Place all analysed data in dictionary
         ## 
         
         data = {}
         data['name'] = name[jn]
         data['color'] = colors[jn]
         
         data['k'] = vk
         data['psd'] = psd_ke
         data['Tk_ke'] = Tk_ke
         data['Pi_ke'] = Pi_ke
         data['Tk_tau'] = Tk_tau
         data['Tk_visc'] = Tk_visc
         if (lrossby and name[jn] == 'ORCA1-N406' and machine == 'jasmin'):
            data['Rossby'] = Ld
         if (lrhines):
            data['Rhines'] = Lr
            
         data['psd_bke'] = psd_bke
         data['psd_beke'] = psd_beke
         data['psd_cke'] = psd_cke
         data['psd_ceke'] = psd_ceke
         data['Tk_bt_bc_bc'] = Tk_bt_bc_bc
         data['Tk_bt_bt_bt'] = Tk_bt_bt_bt
         data['Tk_bc_bc_bc'] = Tk_bc_bc_bc
         
         if (lpsd_freq):
            data['omega'] = vk_t
            data['psd_freq'] = ke1D_t
         
         spectre_list.append(data)
         

if (1):
   if (lke):
      ## kinetic energy
      fig, ax1, ax11 = set_psd_fig()
      ax1.set_ylabel(r'Spectral KE density $[\mathrm{m}^3 \mathrm{s}^{-2}]$')
      
      for data in spectre_list:
         ax1.loglog(data['k'],data['psd'],color=data['color'],label=data['name'],basex=10,basey=10)
      
      draw_lines(ax1,data['k'],data['psd'],m3=True)
      ax11.set_xscale("log") 
      
      ax1.callbacks.connect("xlim_changed", convert_k_to_len)
      ax1.legend(fontsize=10,loc=3,frameon=False)   
      figname = 'all_KE_'+region+'_'+starttime+'-'+endtime
      fig.savefig(pdir+figname+'.pdf',format='pdf')
      
      
   if (lbke):
      ## barotropic kinetic energy
      fig_bke, ax1_bke, ax11_bke = set_psd_fig()
      ax1_bke.set_ylabel('Spectral density of barotropic KE')
      
      for data in spectre_list:
         ax1_bke.loglog(data['k'],data['psd_bke'],color=data['color'],label=data['name'],basex=10,basey=10)
      
      draw_lines(ax1_bke,data['k'],data['psd_cke'],m3=True)
      ax11_bke.set_xscale("log") 
      
      ax1_bke.legend(fontsize=10,loc=3,frameon=False)   
      figname = 'barotropic_KE_'+region+'_'+starttime+'-'+endtime
      fig_bke.savefig(pdir+figname+'.pdf',format='pdf')
      
      
   if (lcke):
      ## baroclinic kinetic energy
      fig, ax1, ax11 = set_psd_fig()
      ax1.set_ylabel('Spectral density of baroclinic KE')
      
      for data in spectre_list:
         ax1.loglog(data['k'],data['psd_cke'],color=data['color'],label=data['name'],basex=10,basey=10)
      
      draw_lines(ax1,data['k'],data['psd_cke'],m3=True)
      ax11.set_xscale("log") 
      
      ax1.callbacks.connect("xlim_changed", convert_k_to_len)
      ax1.legend(fontsize=10,loc=3,frameon=False)      
      figname = 'baroclinic_KE_'+region+'_'+starttime+'-'+endtime
      fig.savefig(pdir+figname+'.pdf',format='pdf')
      
   
   if (lke_freq):
      ax1_psdt.set_ylabel(r'Spectral KE density $[\mathrm{m}^2 \mathrm{s}^{-1}]$')
   
      
   if (lTk_bc_bt):
      ## baroclinic, barotropic interactions
      fig, ax1, ax11, ax2, ax3, ax4 = set_flux_fig()
      
      ax1.set_ylabel(r'bc -> bt $[\mathrm{m}^3 \mathrm{s}^{-3}]$')
      ax2.set_ylabel(r'bt -> bt $[\mathrm{m}^3 \mathrm{s}^{-3}]$')
      ax3.set_ylabel(r'bc -> bc $[\mathrm{m}^3 \mathrm{s}^{-3}]$')
      
      for data in spectre_list:
         vk = data['k']
         v1 = data['Tk_bt_bc_bc']
         v2 = data['Tk_bt_bt_bt']
         v3 = data['Tk_bc_bc_bc']
         
         kmin = vk[0]
         kmax = vk[-1]
         vmax = max( np.abs(v1).max(), np.abs(v2).max(), np.abs(v3).max() )
         vmin = -vmax
         
         if ('Rossby' in data.keys()):
            ax1.semilogx([2*np.pi/data['Rossby'],2*np.pi/data['Rossby']],[vmin,vmax],'-k')
         
         if ('Rhines' in data.keys()):
            ax1.semilogx([2*np.pi/data['Rhines'],2*np.pi/data['Rhines']],[vmin,vmax],color=data['color'],linestyle='-')
         
         plots = []
         labels = []
         
         ## Plot transfer
         p, = ax1.semilogx(vk,v1,label=data['name'],color=data['color'],linestyle='-',basex=10)
         
         plots.append(p)
         labels.append(name[jn])              
         
         ax2.semilogx(vk,v2,color=data['color'],linestyle='-',basex=10)
         
         ax3.semilogx(vk,v3,color=data['color'],linestyle='-',basex=10)
         
      ax1.callbacks.connect("xlim_changed", convert_k_to_len)
      fig.legend(plots,labels,bbox_to_anchor=(0.65,0.7,0.3,0.2), loc=2, borderaxespad=0., fontsize=10)   
      figname = 'bc_bt_KE_transfers_'+region+'_'+starttime+'-'+endtime
      fig.savefig(pdir+figname+'.pdf',format='pdf')
   
   
   if (lTk_wind_visc):
      ax11_ke.set_xlabel(r'Wavelength $2\pi/K$ $[\mathrm{km}]$')
      ax1_ke.set_ylabel(r'Spectral transfer $[\mathrm{m}^3 \mathrm{s}^{-3}]$')
      ax2_ke.set_ylabel(r'Wind $[\mathrm{m}^3 \mathrm{s}^{-3}]$')
      ax22_ke.set_ylabel(r'Viscosity $[\mathrm{m}^3 \mathrm{s}^{-3}]$')
      ax3_ke.set_ylabel(r'Flux $[\mathrm{m}^2 \mathrm{s}^{-3}]$')
   
         
         
plt.show()
