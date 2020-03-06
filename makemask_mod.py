'''
By Hao He, Aug 29th, 2019

'''
import subprocess

def mask_shape(image,region,output):
    makemask(inpimage=image,inpmask=region,mode='copy',output=output)

def mask_threshold(image,lower,upper,output):
    ia.open(image)
    ia.calcmask(image+'>'+str(lower)+' && '+image+'<'+str(upper),name='mask0')
    ia.close()
    makemask(inpimage=image,inpmask=image+':mask0',mode='copy',output=output)
    makemask(mode='delete',inpmask=image+':mask0')

def mask_invert(mask,output):
    ia.open(image)
    ia.calcmask(image+'< 0.5')
    ia.close()
    makemask(inpimage=image,inpmask=image+':mask0',mode='copy',output=output)
    makemask(mode='delete',inpmask=mask+':mask0')

def mask_subtract(mask1,mask2,output):
    immath(imagename=[mask1,mask2],expr='IM0-IM1',outfile='temp.mask')
    ia.open('temp.mask')
    ia.calcmask('"temp.mask">0.5',name='mask0')
    ia.close()
    makemask(mode='copy', inpimage='temp.mask', inpmask='temp.mask:mask0', output=output)
    rmtables('temp.mask')

def mask_and(mask1,mask2,output):
    immath(imagename=[mask1,mask2],expr='IM0+IM1',outfile='temp.mask')
    ia.open('temp.mask')
    ia.calcmask('"temp.mask">1.5',name='mask0')
    ia.close()
    makemask(mode='copy', inpimage='temp.mask', inpmask='temp.mask:mask0', output=output)
    rmtables('temp.mask')
    

Dir='/1/home/heh15/workingspace/Arp240/ratio/'
imageDir=Dir+'NGC5257/1213/combine/'
measureDir=Dir+'NGC5257/test_measure/1213/'
scriptDir=Dir+'script/'

Im12CO10='NGC5257_12CO10_combine_smooth_masked.image.mom0'
Im13CO10='NGC5257_13CO10_12m_smooth_masked.image.mom0'

# copy the image into the current directory
# subprocess.call(['cp','-r',imageDir+Im12CO10,'.'])


image=Im12CO10
lower=2.0; upper=5.0
output='spiralarm_temp.mask'
mask_threshold(image,lower,upper,output)

image=Im12CO10
region='anomaly.crtf'
output='anomaly.mask'
mask_shape(image,region,output)



image=Im12CO10
region='nonarm.crtf'
output='nonarm.mask'
mask_shape(image,region,output)

mask1='spiralarm_temp.mask'
mask2='anomaly.mask'
output='spiralarm_temp1.mask'
mask_subtract(mask1,mask2,output)

mask1='spiralarm_temp1.mask'
mask2='nonarm.mask'
output='spiralarm.mask'
mask_subtract(mask1,mask2,output)

# disk without spiral arm
image=Im12CO10
lower=0.5; upper=2.0
output='disk.mask'
mask_threshold(image,lower,upper,output)

# anomaly with region greater than 2. 
mask1='spiralarm_temp.mask'
mask2='anomaly.mask'
output='anomaly_mod.mask'
mask_and(mask1,mask2,output)

# nonarm with region greater than 2.
mask1='spiralarm_temp.mask'
mask2='nonarm.mask'
output='nonarm_mod.mask' 
mask_and(mask1,mask2,output)

# center of the galaxy
image=Im12CO10
lower=5.0;upper=100
output='center.mask'
mask_threshold(image,lower,upper,output)

# regrid to 12CO21 image
imregrid(imagename='anomaly_mod.mask',template='NGC5257_12CO21_combine_uvtaper_smooth_masked.image.mom0/',output='anomaly_12CO21.fits')

imregrid(imagename='nonarm_mod.mask',template='NGC5257_12CO21_combine_uvtaper_smooth_masked.image.mom0/',output='nonarm_12CO21.fits')
