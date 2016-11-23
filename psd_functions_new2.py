import numpy as np
from netCDF4 import Dataset 
import os,sys,time
from scipy.fftpack import fft, ifft, fftn, ifftn
from scipy.signal import periodogram, hamming, tukey
import scipy.stats as stats
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def find_time(starttime,endtime,nd):
   startday  = int(starttime[6:8])
   startmon  = int(starttime[4:6])
   startyear = int(starttime[0:4])
   
   endday    = int(endtime[6:8])
   endmon    = int(endtime[4:6])
   endyear   = int(endtime[0:4])
   
   ii = 1
   iday  = startday
   imon  = startmon
   iyear = startyear
   itime = iyear * 10000 + imon * 100 + iday
   
   idaymax = [31,28,31,30,31,30,31,31,30,31,30,31]
   
   idays = np.array([iday])
   imons = np.array([imon])
   iyears = np.array([iyear])
   
   if (int(endtime) - int(starttime) >0 ):
      while( itime < endyear * 10000 + endmon * 100 + endday ):
         
         iday = iday + nd
         if (iday > idaymax[imon-1]):
            iday = iday - idaymax[imon-1]
            imon = imon + 1
         
         if (imon > 12):
            imon  = 1
            iyear = iyear + 1
         
         itime = iyear * 10000 + imon * 100 + iday
         print itime
         
         idays = np.append(idays,iday)
         imons = np.append(imons,imon)
         iyears = np.append(iyears,iyear)
      
   print ' First day, month, year ',idays[0],imons[0],iyears[0]
   print ' Last day, month, year ',idays[-1], imons[-1], iyears[-1]
   
   return iyears,imons,idays
   
   
def interp(lst):
   [lon_hi,lat_hi,data_hi,lon_lo,lat_lo] = lst
   
   ## If we use compress the interpolation will fill in masked values, which we do not want.                                                                              
   ## We only flatten, so that interpolation ignores masked values                                                                                                        
   ## Make 1D arrays of lon, lat
   points_hi = np.array([lon_hi.flatten(),lat_hi.flatten()]).transpose()
   data_hi   = data_hi[:,:].flatten()
   
   t0 = time.time()
   data_lo = griddata(points_hi, data_hi, (lon_lo, lat_lo), method='nearest')
   t1 = time.time()
   
   return data_lo
   
   
def read_nemo_grid(hfile,i0,i1,j0,j1):
   print hfile
   nc = Dataset(hfile,'r')
   tlon = nc.variables['glamt'][0,j0:j1,i0:i1]
   tlat = nc.variables['gphit'][0,j0:j1,i0:i1]
   ulon = nc.variables['glamu'][0,j0:j1,i0:i1]
   ulat = nc.variables['gphiu'][0,j0:j1,i0:i1]
   vlon = nc.variables['glamv'][0,j0:j1,i0:i1]
   vlat = nc.variables['gphiv'][0,j0:j1,i0:i1]
   return ulon,ulat,vlon,vlat,tlon,tlat

def read_nemo_zgrid(zfile,i0,i1,j0,j1):
   print zfile
   nc = Dataset(zfile,'r')
   kmt = nc.variables['mbathy'][0,j0:j1,i0:i1]
   dzt = nc.variables['e3t'][0,:,j0:j1,i0:i1]
   dz  = nc.variables['e3t_0'][0,:]
   return dzt,dz,kmt
   

def set_region(region):
   
   ## Jet in double-gyre run
   if (region == 'gyre-jet'):
      lon0 = 0
      lon1 = 16
      lat0 = 19
      lat1 = 31
   
   ## ACC   
   if (region == 'acc-full'):
      lon0 = 30.
      lon1 = 70.
      lat0 = -45
      lat1 = -35
   
   if (region == 'acc-scott'):
      lon0 = -128
      lon1 = -112
      lat0 =  -65
      lat1 =  -49
   
   if (region == 'agulhas-retro'):
      lon0 = 14
      lon1 = 68
      lat0 = -45.5
      lat1 = -37
   if (region == 'agulhas-retro-2'):
      lon0 = 14
      lon1 = 68
      lat0 = -54
      lat1 = -45.5
            
   ## Gulf Stream
   if (region == 'gulfstream-full'):
      lon0 = -75.
      lon1 = -50.
   if (region == 'gulfstream-up'):
      lon0 = -75.
      lon1 = -68.5
   if (region == 'gulfstream-down'):
      lon0 = -68.5
      lon1 = -50.
   if (region[0:10] == 'gulfstream'):
      lat0 = 33.3
      lat1 = 42.
         
   ## Kuroshio
   ## big box
   if (region == 'kuroshio-full'):
      lon0 = 141.5
      lon1 = -176.
   ## downstream
   if (region == 'kuroshio-down'):
      lon0 = 163.
      lon1 = -176 #-165.
   ## upstream
   if (region == 'kuroshio-up'):
      lon0 = 141.5
      lon1 = 163. 
   if (region[0:8] == 'kuroshio'):
      lat0 = 30.0
      lat1 = 40.0
   if (region == 'north-pacific-1'):
      lon0 = 141.5
      lon1 = -176.
      lat0 = 20.0
      lat1 = 32.5
   
   return lon0,lon1,lat0,lat1
   

def read_nemo(ufile,vfile,i0,i1,j0,j1,levels=np.arange(0,75),step=0,\
              plot_ke=False,draw_box=False,aver_type='baroclinic',machine='mac'):
   """
   """
   
   print ufile
   ncu = Dataset(ufile,'r')
   ncv = Dataset(vfile,'r')
         
   ulon = ncu.variables['nav_lon'][j0:j1,i0:i1]
   ulat = ncu.variables['nav_lat'][j0:j1,i0:i1]
   vlon = ncu.variables['nav_lon'][j0:j1,i0:i1]
   vlat = ncu.variables['nav_lat'][j0:j1,i0:i1]
   print ' Read u, v '
   uvel_full = ncu.variables['vozocrtx'][:,levels,j0:j1,i0:i1]
   vvel_full = ncv.variables['vomecrty'][:,levels,j0:j1,i0:i1]
   taux_full = ncu.variables['sozotaux'][:,j0:j1,i0:i1]
   tauy_full = ncv.variables['sometauy'][:,j0:j1,i0:i1]
   
   if ('vozoeivu' in ncu.variables.keys()):
      uvel2 = ncu.variables['vozoeivu'][:,levels,j0:j1,i0:i1]
      uvel_full = uvel_full + uvel2 
      print ' Added GM u velocities '
   if ('vomeeivv' in ncv.variables.keys()):
      vvel2 = ncv.variables['vomeeivv'][:,levels,j0:j1,i0:i1]
      vvel_full = vvel_full + vvel2
      print ' Added GM v velocities '
      
   print ' Mask where u,v == 0 '
   uvel_full = np.ma.masked_where( (uvel_full == 0) & (vvel_full == 0), uvel_full ) 
   vvel_full = np.ma.masked_where( (uvel_full == 0) & (vvel_full == 0), vvel_full )
   taux_full = np.ma.masked_where( (uvel_full[:,0,:,:] == 0) & (vvel_full[:,0,:,:] == 0), taux_full )
   tauy_full = np.ma.masked_where( (uvel_full[:,0,:,:] == 0) & (vvel_full[:,0,:,:] == 0), tauy_full )
         
   print ' u shape: ', uvel_full.shape
            
   if (aver_type == 'barotropic'):
      print ' Calculate vertical mean '
      uvel = np.ma.mean(uvel_full, axis=1)
      vvel = np.ma.mean(vvel_full, axis=1)
   elif (aver_type == 'baroclinic'):
      print ' Select all layers '
      uvel = uvel_full[:,:,:,:]
      vvel = vvel_full[:,:,:,:]
         
   print ' u shape: ', uvel_full.shape
   
   ncu.close()
   ncv.close()
         
   return ulon,ulat,vlon,vlat,uvel,vvel,taux_full,tauy_full


def read_aviso(starttime,endtime,i0,i1,j0,j1,plot_ke=False,draw_box=False,machine='mac'):
   """
   """
   
   if (machine == 'mac'):
      year1 = 2000
      year2 = 2000
   else:
      year1 = 1993
      year2 = 2015
   
   if (machine == 'mac'):
      ufile = '/Users/joakim/Downloads/aviso/dt_global_allsat_msla_uv_%04d0101-%04d1231.nc'
   elif (machine == 'archer'):
      ufile = '/work/n01/n01/joakim2/data/aviso/dt_global_allsat_msla_uv_%04d0101-%04d1231.nc'
   elif (machine == 'jasmin'):
      ufile = '/group_workspaces/jasmin2/aopp/joakim/aviso/dt_global_allsat_msla_uv_%04d0101-%04d1231.nc'
   
   ufile = ufile % (year1,year2)
   
   iyear = np.array([]) ; imon = np.array([]) ; iday = np.array([]) ; itime = np.array([])
   i = 0
   for iyy in range(year1,year2+1):
      daymax = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
      if (np.mod(iyy,4) != 0): 
         daymax[1] = 28
      elif (np.mod(iyy,100) != 0):
         daymax[1] = 29
      elif (np.mod(iyy,400) != 0): 
         daymax[1] = 28
      else:
         daymax[1] = 29
         
      for imm in range(1,13):
         for idd in range(1,daymax[imm-1]+1):
            iyear = np.append(iyear,iyy)
            imon  = np.append(imon, imm)
            iday  = np.append(iday, idd)
            itime = np.append(itime,iyy*10000+imm*100+idd)
            i = i + 1
   
   steps = np.where( (itime >= int(starttime)) & (itime <= int(endtime)) )[0]
   print steps,itime[steps[0]],itime[steps[-1]]
   
   nc = Dataset(ufile,'r')
   lon = nc.variables['lon'][i0:i1]
   lon = np.where( lon>180, lon-360, lon )
   lat = nc.variables['lat'][j0:j1]
   uvel = nc.variables['u'][steps,j0:j1,i0:i1]
   vvel = nc.variables['v'][steps,j0:j1,i0:i1]
   
   nc.close()
   
   lon,lat = np.meshgrid(lon,lat)
   
   return lon,lat,uvel,vvel
   

def rotate_box(xlist,ylist,alpha):
   """
   """
   
   a = alpha*np.pi/180.
   Lx = xlist[1]-xlist[0]
   Ly = ylist[2]-ylist[1]
   
   x1p = xlist[0]
   y1p = ylist[0]
   
   x2p = xlist[0] + np.cos(a) * Lx
   y2p = ylist[0] + np.sin(a) * Lx
   
   x3p = x2p - np.sin(a) * Ly 
   y3p = y2p + np.cos(a) * Ly
   
   x4p = xlist[0] - np.sin(a) * Lx
   y4p = ylist[0] + np.cos(a) * Lx
   
   return [x1p,x2p,x3p,x4p],[y1p,y2p,y3p,y4p] 
   
   
def calculate_vorticity(uvel,vvel,xx,yy):
   """
   """
   
   rot = np.zeros(uvel.shape)
   
   dvdx = (vvel[1:-1,2:]-vvel[1:-1,0:-2]) / (xx[1:-1,2:]-xx[1:-1,0:-2])
   dudy = (uvel[2:,1:-1]-uvel[0:-2,1:-1]) / (yy[2:,1:-1]-yy[0:-2,1:-1])
   
   rot[1:-1,1:-1] = dvdx[:,:] - dudy[:,:]
   
   return rot
   

def calculate_ke(k2D,l2D,uvel,vvel,laverage=True):
   """
   """
   uhat = fftn(uvel)
   vhat = fftn(vvel)
   z = 0.5 * np.real(uhat * np.conj(uhat) + vhat * np.conj(vhat))
   if (laverage):
      nn = (uvel.shape[1]**2 * uvel.shape[0]**2)
      z = z/float(nn)   
   return z

def calculate_ens(k2D,l2D,rot):
   """
   """
   rhat = fftn(rot)
   z = 0.5 * np.real(rhat * np.conj(rhat))
   print rhat.shape
      
   return z

def calculate_spectral_flux(kx,ky,uvel,vvel):
   """
   Calculate spectral flux 
   We assume du/dt = -u du/dx - v du/dy
             dv/dt = -u dv/dx - v dv/dy
   """
   
   uhat = fftn(uvel)
   vhat = fftn(vvel)
   i = np.complex(0,1)
   # du/dx in x,y
   ddx_u = np.real( ifftn(i*kx*uhat) )
   # du/dy in x,y
   ddy_u = np.real( ifftn(i*ky*uhat) )
   # dv/dx in x,y
   ddx_v = np.real( ifftn(i*kx*vhat) )
   # dv/dy in x,y
   ddy_v = np.real( ifftn(i*ky*vhat) )
   
   # adv_u = u * du/dx + v * du/dy
   adv_u = uvel * ddx_u + vvel * ddy_u
   # adv_v = u * dv/dx + v * dv/dy
   adv_v = uvel * ddx_v + vvel * ddy_v
   
   # KE trend from advection: 
   # - u * adv_u - v * adv_v
   # in spectral space
   # The minus sign arises as advection 
   # is on the RHS of the momentum eqs. 
   Tkxky = np.real( -np.conj(fftn(uvel))*fftn(adv_u) - \
                     np.conj(fftn(vvel))*fftn(adv_v) )   #[m2/s3]
   
   return Tkxky


def calculate_spectral_flux_baroclinic_barotropic(kx,ky,u_bt,v_bt,u_bc,v_bc):
   """
   Calculate spectral flux for a triad
   i.e. u_bt * u_bc * du_bc/dx + u_bt * v_bc * du_bc/dy + 
        v_bt * u_bc * dv_bc/dx + v_bt * v_bc * dv_bc/dy
   """
   
   i = np.complex(0,1)
   
   uhat_bt = fftn(u_bt)
   vhat_bt = fftn(v_bt)
   
   nx = u_bc.shape[2]
   ny = u_bc.shape[1]
   nz = u_bc.shape[0]
   
   for jk in range(0,nz):
      uhat_bc = fftn(u_bc[jk,:,:])
      vhat_bc = fftn(v_bc[jk,:,:])
      
      ddx_u_bc = np.real( ifftn(i*kx*uhat_bc) ) # du_bc/dx
      ddy_u_bc = np.real( ifftn(i*ky*uhat_bc) ) # du_bc/dy
      ddx_v_bc = np.real( ifftn(i*kx*vhat_bc) ) # dv_bc/dx
      ddy_v_bc = np.real( ifftn(i*ky*vhat_bc) ) # dv_bc/dy
      
      ddx_u_bt = np.real( ifftn(i*kx*uhat_bt) ) # du_bt/dx
      ddy_u_bt = np.real( ifftn(i*ky*uhat_bt) ) # du_bt/dy
      ddx_v_bt = np.real( ifftn(i*kx*vhat_bt) ) # dv_bt/dx
      ddy_v_bt = np.real( ifftn(i*ky*vhat_bt) ) # dv_bt/dy
      
      if (jk == 0):
         # adv_u = u * du/dx + v * du/dy
         adv_u_bc_bc = u_bc[jk,:,:] * ddx_u_bc + v_bc[jk,:,:] * ddy_u_bc
         adv_u_bt_bt = u_bt[:,:]    * ddx_u_bt + v_bt[:,:]    * ddy_u_bt
         # adv_v = u * dv/dx + v * dv/dy
         adv_v_bc_bc = u_bc[jk,:,:] * ddx_v_bc + v_bc[jk,:,:] * ddy_v_bc
         adv_v_bt_bt = u_bt[:,:]    * ddx_v_bt + v_bt[:,:]    * ddy_v_bt
      else:
         # adv_u = u * du/dx + v * du/dy
         adv_u_bc_bc = adv_u_bc_bc + u_bc[jk,:,:] * ddx_u_bc + v_bc[jk,:,:] * ddy_u_bc
         adv_u_bt_bt = adv_u_bt_bt + u_bt[:,:]    * ddx_u_bt + v_bt[:,:]    * ddy_u_bt
         # adv_v = u * dv/dx + v * dv/dy
         adv_v_bc_bc = adv_v_bc_bc + u_bc[jk,:,:] * ddx_v_bc + v_bc[jk,:,:] * ddy_v_bc
         adv_v_bt_bt = adv_v_bt_bt + u_bt[:,:]    * ddx_v_bt + v_bt[:,:]    * ddy_v_bt
   
   adv_u_bc_bc = adv_u_bc_bc / float(nz)   
   adv_v_bc_bc = adv_v_bc_bc / float(nz)   
   adv_u_bt_bt = adv_u_bt_bt / float(nz)   
   adv_v_bt_bt = adv_v_bt_bt / float(nz)   
   
   # KE trend from advection: 
   # - u * adv_u - v * adv_v
   # in spectral space
   # The minus sign arises as advection 
   # is on the RHS of the momentum eqs. 
   Tk_bt_bc_bc = np.real( -np.conj(uhat_bt)*fftn(adv_u_bc_bc) - \
                           np.conj(vhat_bt)*fftn(adv_v_bc_bc) )   #[m2/s3]
   Tk_bt_bt_bt = np.real( -np.conj(uhat_bt)*fftn(adv_u_bt_bt) - \
                           np.conj(vhat_bt)*fftn(adv_v_bt_bt) )   #[m2/s3]
   
   nn = (nx**2 * ny**2)                        
   data = {}
   data['Tk_bt_bc_bc'] = Tk_bt_bc_bc / float(nn)
   data['Tk_bt_bt_bt'] = Tk_bt_bt_bt / float(nn)
   
   return data


def calculate_spectral_ens_flux(kx,ky,uvel,vvel,rot):
   """
   """
   
   rhat = fftn(rot)
   uhat = fftn(uvel)
   vhat = fftn(vvel)
   i = np.complex(0,1)
   # drot/dx in x,y
   ddx_rot = np.real( ifftn(i*kx*rhat) )
   # drot/dy in x,y
   ddy_rot = np.real( ifftn(i*ky*rhat) )
   
   # adv_rot = u * drot/dx + v * drot/dy
   adv_rot = uvel * ddx_rot + vvel * ddy_rot
   
   # Enstrophy trend from advection: 
   # rot * adv_rot 
   # in spectral space
   Tkxky = np.real( -np.conj(rhat)*fftn(adv_rot) ) 
   
   return Tkxky
   

def calculate_spectral_viscosity(kx,ky,uvel,Ahm,order='4'):
   """
   Calculate spectral flux 
   We assume du/dt = (d4/dx4+d4/dy4) u (order 4) or 
             du/dt = (d2/dx2+d2/dy2) u (order 2) or 
   """
   
   uhat = fftn(uvel)
   i = np.complex(0,1)
   if (order == '4'):
      # d4u/dx4 in x,y
      ddx_u = np.real( ifftn(Ahm * kx**4 * uhat) )
      # d4u/dy4 in x,y
      ddy_u = np.real( ifftn(Ahm * ky**4 * uhat) )
   elif (order == '2'):
      # d2u/dx2 in x,y
      ddx_u = np.real( ifftn(-Ahm * kx**2 * uhat) )
      # d2u/dy2 in x,y
      ddy_u = np.real( ifftn(-Ahm * ky**2 * uhat) )
   
   # KE trend from viscosity: 
   # ddx_u + ddy_u
   # in spectral space
   Tkxky = np.real( np.conj(uhat) * fftn(ddx_u) + \
                    np.conj(uhat) * fftn(ddy_u)   ) 
   
   return Tkxky
   

def calculate_spectral_forcing(kx,ky,uvel,vvel,taux,tauy,rho=1023):
   """
   Calculate spectral flux from wind forcing
   We assume du/dt = 1/rho*(u_conj * taux + v_conj * tauy)
   """
   
   uhat = fftn(uvel)
   vhat = fftn(vvel)
   txhat = fftn(taux/rho)
   tyhat = fftn(tauy/rho)
   i = np.complex(0,1)
   
   # u_conj * taux
   u_taux = np.conj(uhat) * txhat
   # v_conj * tauy
   v_tauy = np.conj(vhat) * tyhat
   
   # KE trend from wind forcing: 
   Tkxky = np.real( u_taux + v_tauy ) 
   
   return Tkxky
   

def integrate_spectrum(psd2D,wvsq,k,dk):
   """
   """
   ## Integrate 2D PSD around lines of constant k
   ## sum(u^2 + v^2) = sum(E)
   ## PSD = d/dk sum(E) [m3/s2]
   ## E = PSD * dk [m2/s2], energy at a given k
   nk = k.shape[0]
   psc = np.zeros((nk))
   for jk in range(0,nk):
      indices = np.where(wvsq >= k[jk]**2)
      psc[jk] = np.sum(psd2D[indices]) 
         
   vpsd   = -(psc[1:] - psc[0:-1]) / dk
   vpsdk  =  (psc[1:] + psc[0:-1]) * 0.5 
   
   return 0.5*(k[1:]+k[0:-1]),vpsd,vpsdk
   
   
def set_wavenumbers_ll(lon0,lon1,lat0,lat1,dlon=0.25,dlat=0.25):
   """
   """
   
   xmin = lon0#np.max(ulon[:,0])
   xmax = lon1#np.min(ulon[:,-1])
   ymin = lat0#np.max(ulat[0,:])
   ymax = lat1#np.min(ulat[-1,:])
   
   # if xmax less than xmin,
   # e.g. xmax = -160 and xmin = 160
   # then add 360 degrees
   if (xmax <= xmin): xmax=xmax+360
   # by checking above, we have 
   # ensured x below is monotonic      
   x = np.arange(xmin,xmax+dlon,dlon)
   # now remove 360 where x > 180 go get 
   # back to original definition
   x = np.where(x>=180,x-360,x)
   
   y = np.arange(ymin,ymax+dlat,dlat)
   xx,yy = np.meshgrid(x,y)
      
   ## Length of domain
   nx = x.shape[0]
   ny = y.shape[0]
   Lx = np.abs(x.max() - x.min())
   Ly = np.abs(y.max() - y.min())
   
   print 'Lx,Ly [deg]',Lx,Ly
   
   ## Wavenumber vectors
   ## Also check if nx is even or odd
   if (np.mod(nx,2) == 0):
      wn_x = 2.0 * np.pi / Lx * \
             np.concatenate( (np.arange(0,nx/2+1),np.arange(-nx/2+1,0)) )
   else:
      nnx = (nx-1)/2
      wn_x = 2.0 * np.pi / Lx * \
             np.concatenate( (np.arange(0,nnx+1),np.arange(-nnx,0)) )
   if (np.mod(ny,2) == 0):   
      wn_y = 2.0 * np.pi / Ly * \
             np.concatenate( (np.arange(0,ny/2+1),np.arange(-ny/2+1,0)) )       
   else:
      nny = (ny-1)/2
      wn_y = 2.0 * np.pi / Ly * \
             np.concatenate( (np.arange(0,nny+1),np.arange(-nny,0)) )        
   
   kx,ky = np.meshgrid(wn_x,wn_y)
   wvsq  = kx**2 + ky**2
   
   #wn_max = np.sqrt( np.max(wn_x)**2 + np.max(wn_y)**2 ) 
   wn_max = np.max( np.sqrt(wvsq) )
   dk = min(2.0*np.pi/Lx,2.0*np.pi/Ly)
   k = np.arange(dk,wn_max+dk,dk)
   nk = k.shape[0]

   return xx,yy,wn_x,wn_y,kx,ky,k,dk
   

def set_wavenumbers_xy(lonmin,lonmax,latmin,latmax,dlon=0.25,dlat=0.25):
   """
   """
   
   
   # if xmax less than xmin,
   # e.g. xmax = -160 and xmin = 160
   # then add 360 degrees
   if (lonmax <= lonmin): lonmax=lonmax+360
   # by checking above, we have 
   # ensured lon below is monotonic      
   lon = np.arange(lonmin,lonmax+dlon,dlon)
   
   lat = np.arange(latmin,latmax+dlat,dlat)
   
   ## Make southwest corner 0
   lon = lon - lonmin
   lat = lat - latmin
      
   ## Length of domain
   nx = lon.shape[0]
   ny = lat.shape[0]
   radius = 6371 * 1000.
   Lx = radius * lon.max() * np.pi / 180. * np.cos(np.mean(lat)*np.pi/180.)
   Ly = radius * lat.max() * np.pi / 180.
   
   dx = radius * dlon * np.pi / 180. * np.cos(np.mean(lat)*np.pi/180.)
   dy = radius * dlat * np.pi / 180.
   
   print 'Lx,Ly [m]',Lx,Ly
   
   x = np.array([0])
   for ji in range(1,nx):
      x = np.append(x,x[ji-1]+dx)
   
   y = np.array([0])
   for jj in range(1,ny): 
      y = np.append(y,y[jj-1]+dy)
   
   xx, yy = np.meshgrid(x,y)
   
   ## Wavenumber vectors
   ## Also check if nx is even or odd
   if (np.mod(nx,2) == 0):
      wn_x = 2.0 * np.pi / Lx * \
             np.concatenate( (np.arange(0,nx/2+1),np.arange(-nx/2+1,0)) )
   else:
      nnx = (nx-1)/2
      wn_x = 2.0 * np.pi / Lx * \
             np.concatenate( (np.arange(0,nnx+1),np.arange(-nnx,0)) )
   if (np.mod(ny,2) == 0):   
      wn_y = 2.0 * np.pi / Ly * \
             np.concatenate( (np.arange(0,ny/2+1),np.arange(-ny/2+1,0)) )       
   else:
      nny = (ny-1)/2
      wn_y = 2.0 * np.pi / Ly * \
             np.concatenate( (np.arange(0,nny+1),np.arange(-nny,0)) )        
   
   kx,ky = np.meshgrid(wn_x,wn_y)
   wvsq  = kx**2 + ky**2
   
   #wn_max = np.sqrt( np.max(wn_x)**2 + np.max(wn_y)**2 ) 
   wn_max = np.max( np.sqrt(wvsq) )
   dk = min(2.0*np.pi/Lx,2.0*np.pi/Ly)
   k = np.arange(dk,wn_max+dk,dk)
   nk = k.shape[0]

   return xx,yy,wn_x,wn_y,kx,ky,k,dk 
   

def smooth_spectrum(data_list,n=20):
   """
   """
   
   data_out = []
   for data in data_list:
      nsmooth = min(data.shape[0],n)
      data = np.convolve(data, np.ones((nsmooth,))/nsmooth, mode='same')
      data_out.append(data)
   
   return data_out
   

def calculate_means_and_anomalies(uvel,vvel,dzt):      
   """
   """
   
   data = {}
   
   if (1):
      ## uvel_vm - vertical mean
      ## uvel_va - deviations from vertical mean
      uvel_vm = np.mean(uvel,axis=0)
      vvel_vm = np.mean(vvel,axis=0)
      uvel_va = np.zeros(uvel.shape)
      vvel_va = np.zeros(vvel.shape)
      for jk in range(0,uvel.shape[0]):
         uvel_va[jk,:,:] = uvel[jk,:,:] - uvel_vm[:,:]
         vvel_va[jk,:,:] = vvel[jk,:,:] - vvel_vm[:,:]
      
      data['u_vm'] = uvel_vm
      data['v_vm'] = vvel_vm
      data['u_va'] = uvel_va
      data['v_va'] = vvel_va
      
      ## uvel_zm - zonal mean
      uvel_zm = np.mean(uvel,axis=2)
      vvel_zm = np.mean(vvel,axis=2)
      ## uvel_za - deviations from zonal mean
      uvel_za = np.zeros(uvel.shape)
      vvel_za = np.zeros(vvel.shape)
      ## uvel_zm_vm - zonal mean of vertical mean
      uvel_zm_vm = np.mean(uvel_vm,axis=1)
      vvel_zm_vm = np.mean(vvel_vm,axis=1)
      ## uvel_za_vm - zonal deviations of vertical mean
      uvel_za_vm = np.zeros(uvel_vm.shape)
      vvel_za_vm = np.zeros(vvel_vm.shape)
      ## uvel_zm_va - zonal mean of vertical deviations
      uvel_zm_va = np.mean(uvel_va,axis=2)
      vvel_zm_va = np.mean(vvel_va,axis=2)
      ## uvel_za_va - zonal deviations of vertical deviations
      uvel_za_va = np.zeros(uvel.shape)
      vvel_za_va = np.zeros(vvel.shape)
      
      for ji in xrange(0,uvel.shape[2]):
         ## zonal anomalies
         uvel_za[:,:,ji] = uvel[:,:,ji] - uvel_zm[:,:]
         vvel_za[:,:,ji] = vvel[:,:,ji] - vvel_zm[:,:]
         ## zonal anomalies of vertical means (barotropic)
         uvel_za_vm[:,ji] = uvel_vm[:,ji] - uvel_zm_vm[:]
         vvel_za_vm[:,ji] = vvel_vm[:,ji] - vvel_zm_vm[:]
         ## zonal anomalies of vertical anomalies (all baroclinic modes)
         uvel_za_va[:,:,ji] = uvel_va[:,:,ji] - uvel_zm_va[:,:]
         vvel_za_va[:,:,ji] = vvel_va[:,:,ji] - vvel_zm_va[:,:]
      
      data['u_zm'] = uvel_zm
      data['v_zm'] = vvel_zm
      data['u_za'] = uvel_za
      data['v_za'] = vvel_za
      data['u_za_vm'] = uvel_za_vm
      data['v_za_vm'] = vvel_za_vm
      data['u_za_va'] = uvel_za_va
      data['v_za_va'] = vvel_za_va
      
   return data   
         