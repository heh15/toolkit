from scipy.ndimage import gaussian_filter
import matplotlib as mpl

def density_contour(
        x, y, weights=None, xlim=None, ylim=None,
        overscan=(0.1, 0.1), logbin=(0.02, 0.02), smooth_nbin=(3, 3),
        levels=(0.393, 0.865, 0.989), alphas=(0.75, 0.50, 0.25),
        color='k', contour_type='contourf', ax=None, **contourkw):
    """
    This function comes from 
    https://github.com/astrojysun/Sun_Plot_Tools/blob/master/sun_plot_tools/ax.py#L101
    Generate data density contours (in log-log space).
    Parameters
    ----------
    x, y : array_like
        x & y coordinates of the data points
    weights : array_like, optional
        Statistical weight on each data point.
        If None (default), uniform weight is applied.
        If not None, this should be an array of weights,
        with its shape matching `x` and `y`.
    xlim, ylim : array_like, optional
        Range to calculate and generate contour.
        Default is to use a range wider than the data range
        by a factor of F on both sides, where F is specified by
        the keyword 'overscan'.
    overscan : array_like (length=2), optional
        Factor by which 'xlim' and 'ylim' are wider than
        the data range on both sides. Default is 0.1 dex wider,
        meaning that xlim = (Min(x) / 10**0.1, Max(x) * 10**0.1),
        and the same case for ylim.
    logbin : array_like (length=2), optional
        Bin widths (in dex) used for generating the 2D histogram.
        Usually the default value (0.02 dex) is enough, but it
        might need to be higher for complex distribution shape.
    smooth_nbin : array_like (length=2), optional
        Number of bins to smooth over along x & y direction.
        To be passed to `~scipy.ndimage.gaussian_filter`
    levels : array_like, optional
        Contour levels to be plotted, specified as levels in CDF.
        By default levels=(0.393, 0.865, 0.989), which corresponds
        to the integral of a 2D normal distribution within 1-sigma,
        2-sigma, and 3-sigma range (i.e., Mahalanobis distance).
        Note that for an N-level contour plot, 'levels' must have
        length=N+1, and its leading element must be 0.
    alphas : array_like, optional
        Transparancy of the contours. Default: (0.75, 0.50, 0.25)
    color : mpl color, optional
        Base color of the contours. Default: 'k'
    contour_type : {'contour', 'contourf'}, optional
        Contour drawing function to call
    ax : `~matplotlib.axes.Axes` object, optional
        The Axes object to plot contours in.
    **contourkw
        Keywords to be passed to the contour drawing function
        (see keyword "contour_type")
    Returns
    -------
    ax : `~matplotlib.axes.Axes` object
        The Axes object in which contours are plotted.
    """
    
    if xlim is None:
        xlim = (10**(np.nanmin(np.log10(x))-overscan[0]),
                10**(np.nanmax(np.log10(x))+overscan[0]))
    if ylim is None:
        ylim = (10**(np.nanmin(np.log10(y))-overscan[1]),
                10**(np.nanmax(np.log10(y))+overscan[1]))

    if ax is None:
        ax = plt.subplot(111)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # force to change to log-log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # generate 2D histogram
    lxedges = np.arange(
        np.log10(xlim)[0], np.log10(xlim)[1]+logbin[0], logbin[0])
    lyedges = np.arange(
        np.log10(ylim)[0], np.log10(ylim)[1]+logbin[1], logbin[1])
    if weights is None:
        hist, lxedges, lyedges = np.histogram2d(
            np.log10(x), np.log10(y),
            bins=[lxedges, lyedges])
    else:
        hist, lxedges, lyedges = np.histogram2d(
            np.log10(x), np.log10(y), weights=weights,
            bins=[lxedges, lyedges])
    xmids = 10**(lxedges[:-1] + 0.5*logbin[0])
    ymids = 10**(lyedges[:-1] + 0.5*logbin[1])
    
    # smooth 2D histogram
    pdf = gaussian_filter(hist, smooth_nbin).T
    
    # calculate cumulative density distribution (CDF)
    cdf = np.zeros_like(pdf).ravel()
    for i, density in enumerate(pdf.ravel()):
        cdf[i] = pdf[pdf >= density].sum()
    cdf = (cdf/cdf.max()).reshape(pdf.shape)

    # plot contourf
    if contour_type == 'contour':
        contourfunc = ax.contour
        contourlevels = levels
    elif contour_type == 'contourf':
        contourfunc = ax.contourf
        contourlevels = np.hstack([[0], levels])
    else:
        raise ValueError(
            "'contour_type' should be either 'contour' or 'contourf'")
    contourfunc(
        xmids, ymids, cdf, contourlevels,
        colors=[mpl.colors.to_rgba(color, a) for a in alphas],
        **contourkw)
    
    return ax

def add_label_band(ax, left, right, label, *, spine_pos=-0.12, tip_pos=-0.09, fontsize=15):
    """
    Helper function to add bracket around x-tick labels.

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to add the bracket to

    left, right : floats
        The positions in *data* space to bracket on the y-axis

    label : str
        The label to add to the bracket

    spine_pos, tip_pos : float, optional
        The position in *axes fraction* of the spine and tips of the bracket.
        These will typically be negative
    fontsize: float
        The fontsize of the label. 

    Returns
    -------
    bracket : matplotlib.patches.PathPatch
        The "bracket" Aritst.  Modify this Artist to change the color etc of
        the bracket from the defaults.

    txt : matplotlib.text.Text
        The label Artist.  Modify this to change the color etc of the label
        from the defaults.

    """
    # grab the yaxis blended transform
    transform = ax.get_xaxis_transform()

    # add the bracket
    bracket = mpatches.PathPatch(
        mpath.Path(
            [
                [left, tip_pos],
                [left, spine_pos],
                [right, spine_pos],
                [right, tip_pos],
            ]
        ),
        transform=transform,
        clip_on=False,
        facecolor="none",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_artist(bracket)

    # add the label
    txt = ax.text(
        (left + right) / 2,
        spine_pos-0.05,
        label,
        ha="center",
        va="center",
        rotation="horizontal",
        clip_on=False,
        transform=transform,
        fontsize=fontsize
    )

    return bracket, txt

def add_scalebar(ax, wcs, length, xy_axis=(0.1,0.8), color='w', linestyle='-', label='',
                 fontsize=12, text_offset=0.1*u.arcsec):
    '''
    Add scale bar to the image, code modified from https://github.com/astropy/astropy-tutorials/issues/443.
    ------
    Parameters:
    ax: matplotlib.axes
        Axes to be plotted on
    wcs: astropy.wcs
        World coordinate system for the image
    length: astropy.Quantity (with units)
        Quantity of length of the scale bar
    xy_axis: tuple
        Coordinate relative to the xy axis
    color: str
        Color for the scale bar
    linestyle, label, fontsize:
        Parameters for plt.plot
    text_offset: astropy.Quantity
        Text vertical offset from the bar
    ------
    Returns:
    lines,txt:
        Line and text object.
    '''
    axis_to_data = ax.transAxes + ax.transData.inverted()
    left_side_pix = axis_to_data.transform(xy_axis)
    left_side = wcs.pixel_to_world(left_side_pix[0],left_side_pix[1])
    lines = ax.plot(u.Quantity([left_side.ra, left_side.ra-length]),
                    u.Quantity([left_side.dec]*2),
                    color=color, linestyle=linestyle, marker=None,
                    transform=ax.get_transform('fk5'),
                   )
    txt = ax.text((left_side.ra-length/2).to(u.deg).value,
                  (left_side.dec+text_offset).to(u.deg).value,
                  label,
                  verticalalignment='top',
                  horizontalalignment='center',
                  transform=ax.get_transform('fk5'),
                  color=color,
                  fontsize=fontsize,
                 )
    return lines,txt

def add_beam(ax, wcs, beam, xy_axis=(0.1,0.1), color='white'):
    '''
    Add the beam to the image
    -----
    Parameters:
    ax: matplotlib.axes
        Axes to be ploted
    wcs: astropy.wcs
        World Coordinate System for the image
    beam: radio-beam.Beam
        Beam object
    xy_axis: coordinate relative to xy axis
    '''
    axis_to_data = ax.transAxes + ax.transData.inverted()
    xcen_pix, ycen_pix = axis_to_data.transform(xy_axis)
    pixscale = np.sqrt(np.sum(wcs.wcs.cdelt**2))*u.deg
    ellipse_artist = beam.ellipse_to_plot(xcen_pix, ycen_pix, pixscale)
    ellipse_artist.set_color(color)
    _ = ax.add_artist(ellipse_artist)

    return

def add_colorbar(ax, orientation='horizontal', location='top', label=''):
    '''
    Add color bar to image with astropy coordinates. 
    (Note: this function has only been tested for adding colorbar at the top so far.)
    ------
    Parameters:
    ax: matplotlib.axes
        The axes to add colorbar.
    orientation: 'horizontal' or 'vertical'.
        Colorbar orientation. 
    location: 'top', 'bottom', 'left', 'right'
        At which side of the image to add the color bar. 
    label: strs
        The color bar label
    ------
    Return:
    cbar: matplotlib.colorbar
    '''
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation=orientation)
    cax.coords[0].set_axislabel(label, fontsize=15)
    cax.coords[0].set_axislabel_position('t')
    cax.coords[0].set_ticklabel_position('t')
    cax.coords[0].set_ticks_position('t')
    cax.coords[1].set_axislabel(' ')
    cax.coords[1].set_ticks_visible(False)
    cax.coords[1].set_ticklabel_visible(False)
    cax.coords.grid(draw_grid=False)

    return cbar

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    Truncate the colormap in matplotlib, modified from 
    https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    ------
    Parameters:
    cmap: 
        Python color maps
        > cmap = plt.get_cmap('jet')
    minval, maxval: float
    n: int
    ------
    Return:
    newcmap: 
        truncated color map
    '''
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def combine_colormap(colors1, colors2)
    '''
    Combine 2 color maps together
    -------
    Parameters:
    colors1, colors2: numpy array 
        Two colors that are going to be used
        > colors1 = plt.cm.binary(np.linspace(0., 1, 128))
        > colors2 = plt.cm.gist_heat(np.linspace(0, 1, 128))
    ------
    Returns:
    mymap: 
        The newly defined cmap
    '''
    colors_combined = np.vstack((colors1, colors2))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colors_combined)

    return mymap
