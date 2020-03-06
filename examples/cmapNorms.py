'''
By Nathan Brunetti

This is a place to store custom matplotlib color map normalizers. brunettn uses
this in imshow for plotting 2D data, like maps of the sky.
'''

import matplotlib.colors as colors
import numpy as np

class CASApower(colors.Normalize):
    """matplotlib Normalize object to replicate CASA 'power scaling' of colors.

    Parameters
    ----------
    vmin : float, optional
        Data value to map to the minimum of the color map, along with values
        below vmin. Default is the minimum data value.

    vmax : float, optional
        Data value to map to the maximum of the color map, along with values
        above vmax. Default is the maximum data value.

    power : float, optional
        Controls the logarithmic scaling of data values to the color map. Can
        be positive or negative. Default is 0 (linear scaling).

    See the CASA documentation on the Viewing Images and Cubes page
    (e.g. https://casa.nrao.edu/casadocs/casa-5.4.1/image-cube-visualization/viewing-images-and-cubes)
    about scaling power cycles for details on how this mapping is calculated. 
    """
    def __init__(self, vmin=None, vmax=None, power=0):
        self.power = power
        colors.Normalize.__init__(self, vmin, vmax, True)

    def __call__(self, value, clip=None):
        if self.power < 0.0:
            x = [self.vmin, self.vmax]
            y = [1, 10**np.abs(self.power)]
            step1 = np.interp(value, x, y)
            step2 = np.log10(step1)
            x = [0, np.abs(self.power)]
            y = [0, 1]
        else:
            x = [self.vmin, self.vmax]
            y = [0.0, self.power]
            step1 = np.interp(value, x, y)
            step2 = 10**step1
            x = [1, 10**self.power]
            y = [0, 1]
        return np.ma.masked_invalid(np.interp(step2, x, y))
