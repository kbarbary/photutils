# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math

import numpy as np

def background(data, mask=None, boxsize=None, box_adjust='last',
               cenfunc=np.median, varfunc=np.var, smoothing=None):
               #mode_estimator=(2.5, 1.5):
    """Estimate global background and standard deviation of image data.

    Currently, only 2-d arrays are supported.
    
    Parameters
    ----------
    data : array_like
        The array on which to estimate the background.
    boxsize : int, optional
        The size of boxes in pixels. Can be a tuple (width x height) for
        rectangles. If None, a scalar (non-variable) background is returned.
    box_adjust : {'all', 'last', 'none'}
        How to adjust boxes if the data shape is not an exact multiple of
        boxsize:
        
        'all'
            Evenly space all boxes, making them as close to `boxsize` as
            possible.
        'last'
            Adjust last box: if less than half of `boxsize`, combine with
            previous box. Otherwise leave as own box.
        'none'
            No adjustment. Last box will be datashape % boxsize.

    method : {'mean', 'median', 'mode', 'clipmean',
        'clipmedian', 'clipmode', 'hybrid'}, optional
        Method used to estimate background in each box.
    mode_estimator: sequence, optional
        For methods that involve the mode, the mode is estimated with
        mode = mode_estimator[0] * median - mode_estimator[1] * mean
    smoothing: {''}
        How to interpolate between boxes. If ``None``, do not interpolate.
    mask : array_like, bool, optional
        Pixels to ignore when estimating background. Must be same shape
        as `data`.

    Returns
    -------
    bkg : float or `~numpy.ndarray`
    std : float or `~numpy.ndarray`
    """

    # Check dimensionality of data
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("only 2-d arrays supported.")

    # Check mask
    if mask is not None:
        mask = np.asarray(mask).astype(np.bool)
        if mask.shape != data.shape:
            raise ValueError("shapes of mask and data must match")

    # If boxsize is None, set it to the shape of the data so that a single
    # box contains all the data. 
    if boxsize is None:
        boxsize = data.shape

    # Make boxsize a 1-d array of ints.
    boxsize = np.atleast_1d(boxsize).astype(np.int)
    if boxsize.ndim != 1:
        raise ValueError("Boxsize can be at most 1-dimensional")
    if boxsize.shape[0] not in [1, 2]:
        raise ValueError("boxsize must be length 1 or 2.")

    # Broadcast boxsize to match data.shape.
    # (`datashape` is not used, this is just for broadcasting boxsize.)
    datashape, boxsize = np.broadcast_arrays(data.shape, boxsize)
    
    # Define bin_edges
    if box_adjust == 'none':
        bin_edges = [np.arange(0, data.shape[i] + boxsize[i], boxsize[i])
                     for i in [0, 1]]
        for i in [0, 1]:
            bin_edges[i][-1] = data.shape[i]

    elif box_adjust == 'last':
        bin_edges = []
        for i in [0, 1]:
            remainder = data.shape[i] % boxsize[i]
            if remainder == 0 or remainder > boxsize[i] / 2.:
                bin_edges.append(np.arange(0, data.shape[i] + boxsize[i],
                                           boxsize[i]))
            else:
                bin_edges.append(np.arange(0, data.shape[i], boxsize[i]))
            bin_edges[i][-1] = data.shape[i]
    elif box_adjust == 'all':
        # TODO: implement this
        raise ValueError('box_adjust=all not yet implemented')
    else:
        raise ValueError('Unrecognized value for box_adjust: {0}'.format(
                box_adjust))

    # Initialize background array
    # TODO: make the dtype match data.dtype?
    mesh_shape = [bin_edges[0].shape[0] - 1, bin_edges[1].shape[0] - 1]
    if smoothing is not None:
        bkg_mesh = np.empty(mesh_shape, dtype=np.float)
        var_mesh = np.empty(mesh_shape, dtype=np.float)
    bkg = np.empty(data.shape, dtype=np.float)
    std = np.empty(data.shape, dtype=np.float)


    # Loop over output boxes
    for j in range(mesh_shape[0]):
        for i in range(mesh_shape[1]):
            box_slice = [slice(bin_edges[j], bin_edges[j + 1]),
                         slice(bin_edges[i], bin_edges[i + 1])]
            subdata = data[box_slice]
            if mask is not None:
                subdata = subdata[mask[box_slice]]
            
            box_cen = cenfunc(subdata)
            box_var = varfunc(subdata)
            if smoothing is None:
                bkg[box_slice] = box_cen
                std[box_slice] = np.sqrt(box_var)
            else:
                bkg_mesh[j, i] = box_cen
                var_mesh[j, i] = box_var

    if smoothing is None:
        return bkg, std
    else:
        # TODO implement smoothing
        print "smoothing not yet implemented"
