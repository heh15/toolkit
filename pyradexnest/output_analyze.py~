import os
import sys
sys.path.insert(0,os.path.abspath('.'))
'''
Hao He: add the current directory to the system directory
'''

import pymultinest
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
import os
import cPickle as pickle
from astropy.table import Table, Column
import sys
from pyradexnest_analyze_tools import *
from config import *


simplify=True

#### LOAD THE STATS
title=os.getcwd()
title=title.split('/')[-1]

### Backwards compatibility. Previously, config file just had n_dims.
# Now, config file has n_comp and n_mol, from which n_dims is calculated (also included)
try:
    n_mol
except:
    n_mol=1
    n_comp=1 if n_dims==4 else 2
    
# Total number of parameters: Dimensions, "Secondary Parameters", SLED Likelihoods
if n_comp==2:
    n_sec=[6,3]
    n_sled=2*sled_to_j*n_mol
else:
    n_sec=[3]
    n_sled=sled_to_j*n_mol
n_params =n_dims + np.sum(n_sec) + n_sled

meas=pickle.load(open("measdata.pkl","rb"))
lw=np.log10(meas['head']['lw'])
# If meas doesn't include tbg, just the old default, 2.73 K
if 'tbg' not in meas: meas['tbg']=2.73
# If not calculated using multimol, won't have secmol.
if 'secmol' not in meas['head']: meas['head']['secmol']=[]

a = pymultinest.Analyzer(n_params = n_params)
s = a.get_stats()
data= a.get_data()
# Check if a.post_file exists; this separates the data by mode.
#### TEMPORARY FIX, in case old version of pymultinest with typo is being used.
if a.post_file==u'chains/1-post_seperate.dat': a.post_file=u'chains/1-post_separate.dat'
if os.path.isfile(a.post_file):
    datsep=post_sep(a.post_file)  # Divide the "data" up by mode.
else:
    datsep={}
    datsep[0]=data
datsep['all']=data
bestfit=a.get_best_fit()
cube=bestfit['parameters'] # The best fit is not the same as the mode, cube=s['modes'][0]['maximum']
nmodes=len(s['modes'])

#### PLOT SETTINGS
# Get the correct plot colors, factors to add, indices to use, etc.
[parameters,add,mult,colors,plotinds,sledcolors]=define_plotting_multimol(n_comp,n_mol,n_dims,n_sec,n_params,sled_to_j,lw)
modecolors=['g','m','y','c','k','r','b']
for x in plotinds: print [parameters[y] for y in x] # A quick check of plot indices, 6/2/2016
    
nicenames=[r'log(n$_{H2}$ [cm$^{-3}$])',r'log(T$_{kin}$ [K])',r'log(N$_{CO}$ [cm$^{-2}$])',r'log($\Phi$)',
           r'log(L[erg s$^{-1}$])',r'log(Pressure [K cm$^{-2}$])',r'log(<N$_{CO}$> [cm$^{-2}$]',
           r'log(Ratio L$_{warm}$/L$_{cold}$)',r'log(Ratio P$_{warm}$/P$_{cold}$)',r'log(Ratio <N>$_{warm}$/<N>$_{cold}$)']
if sled_to_j:
    for x in range(sled_to_j): nicenames.append(r'Flux J='+str(x+1)+'-'+str(x)+' [K]')
# Insert "nicenames" if multimol. Preserve correct order! Fixed 6/2/2016
for i,secmol in enumerate(meas['head']['secmol']):
    tmp=nicenames.insert(4+i,r'X$_{'+secmol+'/'+meas['head']['mol']+'}$')

# Check if we need to fix a completely absurd modemean in the fluxes from radex
if sled_to_j: 
    fix_flux_modemean(s,datsep,plotinds)
    # Addition, do this for the secondary molecules as well. 6/2/2016
    for i,secmol in enumerate(meas['head']['secmol']):
        print [parameters[y] for y in plotinds[5+i]]
        fix_flux_modemean(s,datsep,plotinds,useind=5+i)

# Determine plot xrange from the prior limits.
xrange=np.ndarray([n_dims,2])
xrange[:,0]=0
xrange[:,1]=1
myprior(xrange[:,0],n_dims,n_params)
myprior(xrange[:,1],n_dims,n_params)

# Squash it down if we have 2 components.
if n_comp==2:
    for i in range(4):
        xrange[i,0]=min(xrange[i,0],xrange[i+4,0])
        xrange[i,1]=max(xrange[i,1],xrange[i+4,1])
    for i in range(n_mol-1):
        xrange[i+8,0]=min(xrange[i+8,0],xrange[i+8+n_mol-1,0])
        xrange[i+8,1]=max(xrange[i+8,1],xrange[i+8+n_mol-1,1])
    # Okay I got lazy here.
    if n_mol==2: 
        xrange=xrange[[0,1,2,3,8]]
    elif n_mol==3:
        xrange=xrange[[0,1,2,3,8,10]]
    elif n_mol==4:
        xrange=xrange[[0,1,2,3,8,10,12]]

# Add linewidth to column density range
xrange[2,:]+=lw

######################################

# If a binned pickle already exists and is more 
#   recent than chains, use it.
# This is because binning takes time, and you don't want to 
# redo it if you are just replotting.
# Otherwise, do all the binning and save it to a pickle.

distfile='distributions.pkl'
dists=get_dists(distfile,s,datsep,n_dims + np.sum(n_sec),grid_points=40)

# Sanity check on the distributions.
nrow,ncol,unused=nbyn(n_params)
fig,axarr=plt.subplots(nrow,ncol,num=0,sharex=False,sharey=False,figsize=(4*ncol,4*nrow))
for x in np.arange(0,n_params):
    ind=np.unravel_index(x,axarr.shape)
    axarr[ind].plot(dists['all'][x][0],dists['all'][x][1],color=colors[x])
    axarr[ind].set_xlabel(parameters[x])
for i in unused:
    ind=np.unravel_index(i,axarr.shape)
    axarr[ind].axis('off')
fig.tight_layout()
fig.savefig('fig_raw.png')
print 'Saved fig_raw.png'
plt.close()

######################################
# Table.

table=maketables(s,n_params,parameters,cube,add,mult,n_comp,title=title,addmass=meas['addmass'],n_dims=n_dims)
modemed=table['Mode Mean'][0:n_params]-add      # Mass is last; not included.
modemax=table['Mode Maximum'][0:n_params]-add
modesig=table['Mode Sigma'][0:n_params]
pickle.dump(table, open('table.pkl', "wb") )


############################################################
####  plot the SLED
from matplotlib.ticker import MaxNLocator

mol=meas['head']['mol']
ok=np.all([meas['flux']!=0,meas['mol']==mol],axis=0)#[meas['flux']!=0]
ul=np.all([meas['flux']==0,meas['mol']==mol],axis=0)#[meas['flux']==0]

## fill the sled (from modemed)

sledinds=[8,9,10]
yfill1=modemed[sledinds]-modesig[sledinds]
yfill2=modemed[sledinds]+modesig[sledinds]  
yfill1[yfill1 <= 0]=1e-10
yfill2[yfill2 <= 0]=2e-10

# fig=plt.figure()
# ax=fig.gca()
# ax.fill_between(range(1,4), yfill1, yfill2, where=yfill2>=yfill1, facecolor='gray', interpolate=True)
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.show()

## best fit line (from modemed)
logcol1=cube[2]
dat=pyradex.pyradex(minfreq=1, maxfreq=1600,temperature=np.power(10,cube[1]), column=np.power(10,logcol1), collider_densities={'H2':np.power(10,cube[0])},tbg=meas['tbg'], species=mol, velocity_gradient=1.0, debug=False, return_dict=True)
dat['FLUX_Kkms']=np.array(map(float,dat['FLUX_Kkms']))*np.power(10,cube[3])
model1=dat['FLUX_Kkms']
ladder=dat['J_up']
ladder=map(int,ladder)

# fig=plt.figure()
# plt.plot(dat['J_up'],model1,color='blue',marker=None)
# plt.show()

## Data points from the measurement (from meas)
xplot=meas['J_up']
yplot=meas['flux']
yerr=meas['sigma']

# fig=plt.figure()
# ax=fig.gca()
# ax.fill_between(range(1,4), yfill1, yfill2, where=yfill2>=yfill1, facecolor='gray', interpolate=True)
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.errorbar(xplot[ok],yplot[ok],yerr[ok],color='black')
# plt.plot(dat['J_up'],model1,color='blue',marker=None)
# plt.show()

### 13CO molecules

mol2=meas['head']['secmol'][0]
ok2=np.all([meas['flux']!=0,meas['mol']==mol2],axis=0)#[meas['flux']!=0]
ul2=np.all([meas['flux']==0,meas['mol']==mol2],axis=0)#[meas['flux']==0]

## fill the sled (from modemed)

sledinds=[11,12,13]
yfill1_13co=modemed[sledinds]-modesig[sledinds]
yfill2_13co=modemed[sledinds]+modesig[sledinds]  
yfill1_13co[yfill1_13co <= 0]=1e-10
yfill2_13co[yfill2_13co <= 0]=2e-10

# fig=plt.figure()
# ax=fig.gca()
# ax.fill_between(range(1,4), yfill1, yfill2, where=yfill2>=yfill1, facecolor='gray', interpolate=True)
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.show()

## best fit line (from modemed)
logcol1=cube[2]+cube[4]
dat=pyradex.pyradex(minfreq=1, maxfreq=1600,temperature=np.power(10,cube[1]), column=np.power(10,logcol1), collider_densities={'H2':np.power(10,cube[0])},tbg=meas['tbg'], species=mol2, velocity_gradient=1.0, debug=False, return_dict=True)
dat['FLUX_Kkms']=np.array(map(float,dat['FLUX_Kkms']))*np.power(10,cube[3])
model1_13co=dat['FLUX_Kkms']
ladder_13co=dat['J_up']
ladder_13co=map(int,ladder_13co)

# fig=plt.figure()
# plt.plot(dat['J_up'],model1,color='blue',marker=None)
# plt.show()

## Data points from the measurement (from meas)
xplot=meas['J_up']
yplot=meas['flux']
yerr=meas['sigma']

### plot both molecules

fig=plt.figure()
ax=fig.gca()
ax.set_yscale('log',nonposy='clip')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.errorbar(xplot[ok],yplot[ok],yerr[ok],color='black',marker="o",label='12CO data')
plt.plot(ladder,model1,color='blue',marker=None,label='model')
plt.errorbar(xplot[ok2],yplot[ok2],yerr[ok2],color='black',marker="s",label='13CO data')
plt.plot(ladder_13co,model1_13co,color='blue',marker=None)
plt.ylim(0.01,1.0)
# plt.xlim(None,8)
plt.ylabel('K (per km/s)', size=20)
plt.xlabel('Upper J', size=20)
plt.legend(fontsize=15)
ax.tick_params(labelsize=15)
fig.tight_layout()
plt.savefig('fig_sled.png')

############################################################
#### plot the temperature, density contour map
# mode is from s

i=1;j=0
# mode=s['modes'][0]
presscontour=[3,4,5,6,7,8]
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_ylabel(r'$\log_{10}(T_{kin})$ (K)', size=20)
ax.set_xlabel('$\log_{10}(n_{H2})$ (cm$^{-3}$)', size=20)
ax.contourf(dists['all'][i,j][:,:,1]*mult[j], dists['all'][i,j][:,:,0]*mult[i]+add[i], dists['all'][i,j][:,:,2],5, cmap=cm.gray_r, alpha = 0.8,origin='lower') 
ax.axvline(x=cube[j]*mult[j]+add[j],color='k',linestyle='--',label='4D Max')
ax.axhline(y=cube[i]*mult[i]+add[i],color='k',linestyle='--')
ax.axvline(x=modemed[j]*mult[j]+add[j],color='green',label='Mode Mean')
ax.axhline(y=modemed[i]*mult[i]+add[i],color='green')
for p in presscontour:
    ax.plot([ax.set_xlim()[0],p-ax.set_ylim()[0]],
            [p-ax.set_xlim()[0],ax.set_ylim()[0]],':k')
    ax.annotate('{:.0f}'.format(p),xy=(p-ax.set_ylim()[1],ax.set_ylim()[1]),xycoords='data', fontsize=15)
ax.tick_params(labelsize=15)
plt.legend(fontsize=15)
fig.tight_layout(pad=1.5)
plt.savefig('fig_contour.png')


############################################################
#### plot the column density and beam filling factor contour map
i=3;j=2
mode=s['modes'][0]
bacdcontour=[15,16,17,18,19,20,21,22,23]
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_ylabel('$\log_{10} (\eta_{bf})$', fontsize=20)
ax.set_xlabel('$\log_{10}(N_{12CO})$ (cm$^{-2}$)', fontsize=20)
ax.contourf(dists['all'][i,j][:,:,1]*mult[j]+add[j], dists['all'][i,j][:,:,0]*mult[i]+add[i], dists['all'][i,j][:,:,2],5, cmap=cm.gray_r, alpha = 0.8,origin='lower') 
ax.axvline(x=cube[j]*mult[j]+add[j],color='k',linestyle='--',label='4D Max')
ax.axhline(y=cube[i]*mult[i]+add[i],color='k',linestyle='--')
ax.axvline(x=modemed[j]*mult[j]+add[j],color='green',label='Mode Mean')
ax.axhline(y=modemed[i]*mult[i]+add[i],color='green')
for p in bacdcontour: 
    ax.plot([ax.set_xlim()[0],p-ax.set_ylim()[0]],
                        [p-ax.set_xlim()[0],ax.set_ylim()[0]],':k')
    ax.annotate('{:.0f}'.format(p),xy=(p-ax.set_ylim()[1],ax.set_ylim()[1]),xycoords='data', fontsize=15)
ax.tick_params(labelsize=15)
plt.legend(fontsize=15)
fig.tight_layout(pad=1.5)
plt.savefig('fig_contour2.png')
