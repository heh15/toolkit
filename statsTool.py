def Random_Points_in_Polygon(polygon, number):
    '''
    Generate random points in 2D polygons
    ------
    Parameters:
    polygon: shapely.Polygon
        The polygon region in which the points are
        generated. 
    number: int
        The number of points to be generated
    ------
    Return:
    ras, decs: 
        1d array of RA and Dec in degree. 
    ------
    Examples:
    > from shapely.geometry import Point, Polygon
    > # covert ds9 regions to polygon region. 
    > vert_ra = np.array(Antennae_reg.vertices.ra)
    > vert_dec = np.array(Antennae_reg.vertices.dec)
    > verts = np.transpose(np.vstack((vert_ra, vert_dec)))
    > Antennae_polygon = Polygon(verts)
    > # generate random points
    > ras, decs = Random_Points_in_Polygon(Antennae_polygon, number)
    '''
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)

    ras = np.array([point.x for point in points]); decs = np.array([point.y for point in points])

    return ras, decs

def calc_2pCorr_xi(RA, Dec, randRA, randDec, RA1, Dec1, randRA1, randDec1):
    '''
    Calculate the two-point cross correlation function
    ------
    Parameters: 
    RA, Dec: 
        Coordinates for the first catalogs in degree
    randRA, randDec:
        Coordinates for the first random catalogs in degree
    RA1, Dec1:
        Coordinates for the second catalogs in degree
    randRA1, randDec1:
        Coordinates for the second random catalogs
    ------
    Returns:
    logr, xi, varxi: list
        logr is the list of radius in natural logrithm, xi is the correlation
        coefficient, varxi is the error. 
    ------
    Examples: 
    > import treecorr
    > logr, xi, varxi = calc_2pCorr_xi(RA, Dec, randRA, randDec, RA1, Dec1, randRA1, randDec1)
    '''
    # counting the pairs for real catalogs
    cat = treecorr.Catalog(ra=RA, dec=Dec, ra_units='deg', dec_units='deg')
    cat1 = treecorr.Catalog(ra=RA1, dec=Dec1, ra_units='deg', dec_units='deg')
    nn = treecorr.NNCorrelation(min_sep=0.5, max_sep=100.0, bin_size=0.25, sep_units='arcsec',
                                bin_type='Log')
    nn.process(cat, cat1)
    # counting the pairs for random catalogs
    randcat = treecorr.Catalog(ra=randRA, dec=randDec, ra_units='deg', dec_units='deg')
    randcat1 = treecorr.Catalog(ra=randRA1, dec=randDec1, ra_units='deg', dec_units='deg')
    rr = treecorr.NNCorrelation(min_sep=0.5, max_sep=100.0, bin_size=0.25, sep_units='arcsec',
                                bin_type='Log')
    rr.process(randcat, randcat1)
    # correlation for random and real pairs
    dr = treecorr.NNCorrelation(min_sep=0.5, max_sep=100.0, bin_size=0.25, sep_units='arcsec',
                               bin_type='Log')
    dr.process(cat, randcat1)
    rd = treecorr.NNCorrelation(min_sep=0.5, max_sep=100.0, bin_size=0.25, sep_units='arcsec',
                               bin_type='Log')
    rd.process(randcat, cat1)
    # calculate the two-point correlation function
    xi, varxi = nn.calculateXi(rr=rr, dr=dr, rd=rd)

    return nn.logr, xi, varxi

def pyvorCells_to_regionPix(cells):
    '''
    Convert the Voronoi tessellation object to a list of regionPix objects
    ------
    Parameters:
    '''
    regions_pix = []
    for cell in cells:
        vertices = np.array(cell['vertices'])
        vertices_pix = PixCoord(x=vertices[:,0], y=vertices[:,1])
        region_pix = PolygonPixelRegion(vertices=vertices_pix)
        regions_pix.append(region_pix)
    regions_pix = np.array(regions_pix)
    
    return regions_pix
