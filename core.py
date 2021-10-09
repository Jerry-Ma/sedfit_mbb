#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-07-11 18:04
# Python Version :  2.7.11
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
core.py

"""

from __future__ import print_function
import sys
import logging
import time
from datetime import timedelta
import numpy as np
import numpy.lib.recfunctions as rfn
import lmfit
import os

from . import mbb
from .plot import plot_fit

# from mympl import SolarizedColor as sol
from .mympl import TextStyle as tex
from . import mympl


NAME = 'SEDFIT'

logging.basicConfig(format='[%(name)s] %(message)s', level=logging.DEBUG)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


class SEDFitConf(object):
    """interfacing the input config module"""

    filedefs = [
            ('cat_in', None), ('cat_out', None),
            ('temp_out', None),
            ]
    seddefs = [
            ('id_col', None), ('z_col', None), ('t_col', 't'),
            ('sed_wls', None),
            ('sed_flux_cols', None), ('sed_err_cols', None),
            ]
    fitdefs = [
            ('lambda0', 100.), ('beta', 1.5),
            ]
    plotdefs = [
            ('leg_name', True),
            ('leg_z_fmt', "{:s}"),
            ('leg_t_fmt', "{:s}"),
            ('leg_loc', "upper left"),
            ]
    runtime = [
        ('unpack', True), ('unpack_dir', 'temp_unpacked'),
        ]

    def __init__(self, config):
        missingkeyerror = "missing key in config module: {0}"
        for key, defval in self.filedefs + self.seddefs + \
                self.fitdefs + self.plotdefs + self.runtime:
            try:
                val = getattr(config, key)
            except AttributeError:
                if defval is None:
                    raise RuntimeError(missingkeyerror.format(key))
                elif isinstance(defval, str):
                    val = defval.format(**self.__dict__)
                else:
                    val = defval
            setattr(self, key, val)

    def pprint(self):
        result = []
        for la, label in [('filedefs', 'Files'), ('seddefs', 'SED Defs'),
                          ('fitdefs', 'Fitting Params'),
                          ('plotdefs', 'Plotting Params'),
                          ('runtime', 'Runtime Params')]:
            section = ['{0:>10s}{1:^20s}{2:<50s}'
                       .format('.' * 10, '[ {0} ]'.format(label), '.' * 50), ]
            for k, _ in getattr(self, la):
                v = getattr(self, k)
                section.append('{0:>14s}{1:^4s}{2:<62s}'
                               .format(k, ':', str(v)))
            result.append('\n'.join(section))
        return 'config load from {0}:\n'.format(
                sys.modules['__main__'].__file__) + \
            '\n'.join(result) + '.' * 80


def configure(config, args):
    """handles command line arguments, validates the input configuration,
    returns SEDFitConf object"""

    sfconf = SEDFitConf(config)
    # TODO implement command line interface
    option = {}
    if len(args) > 0:
        if args[-1] == 'save':
            args.pop()
            option['is_save'] = True
        else:
            option['is_save'] = False
    option['args'] = args
    return sfconf, option


def is_fitting_done(config):
    if not os.path.isfile(config.cat_out):
        return False
    in_time = os.path.getmtime(config.cat_in)
    out_time = os.path.getmtime(config.cat_out)
    return out_time > in_time


def bootstrap():
    """
    entry point; parse command line argument, create data object,
    and run the fitting
    """
    logger = logging.getLogger(NAME)
    print("+- FIR SED fitting powered by LMFIT ver. {0}  -+ "
          " +- Author: Jerry Ma -+"
          .format(lmfit.__version__))
    config, option = configure(sys.modules['__main__'], sys.argv[1:])
    if option['args']:  # option by default is glob string for plotting
        if not is_fitting_done(config):
            raise RuntimeError('fitting has not been done yet, '
                               'try run script without command line arguments')
        else:
            batch_plot_sed(config, option)
    else:
        logger.info(config.pprint())
        batch_fit_mbb(config, option)


def batch_fit_mbb(config, option):

    # load object seds
    logger = logging.getLogger(NAME)
    cat_in = np.atleast_1d(
            np.genfromtxt(config.cat_in, names=True, dtype=None,
                          encoding="utf-8"))

    ids = cat_in[config.id_col]
    zs = cat_in[config.z_col]
    wls = np.array(config.sed_wls)
    seds = cat_in[config.sed_flux_cols]
    errs = cat_in[config.sed_err_cols]
    # optional column
    if config.t_col in cat_in.dtype.names:
        ts = cat_in[config.t_col]
    else:
        ts = None

    logger.info('input catalog: {0:s}'.format(config.cat_in))
    logger.info('# of objects: {0:d}'.format(len(cat_in)))
    logger.info('# of bands: {0:d}'.format(len(wls)))

    # initiate mbb model fitter
    fitter = mbb.MBB(beta=config.beta, lam0=config.lambda0)

    # do the fitting
    timestamp = time.time()
    bestfit = fitter.batch_fit(wls, ids, seds, errs, zs, ts=ts, verbose=False)
    elapsed_time = timedelta(seconds=time.time() - timestamp)
    logger.info('finished @ {0}'.format(elapsed_time))
    # save the results
    result = rfn.merge_arrays([cat_in, bestfit], flatten=True, usemask=False)
    np.savetxt(config.cat_out, result, header=' '.join(result.dtype.names),
               fmt='%s', encoding='utf-8')
    logger.info('result saved to {0}'.format(config.cat_out))


def wrap_if_long(string, n):
    if n < 7:
        return string
    elif len(string) > n:
        return string[:n - 5] + '...' + string[-2:]
    else:
        return string


def batch_plot_sed(config, option):
    logger = logging.getLogger(NAME)
    cat_out = np.atleast_1d(
        np.genfromtxt(config.cat_out, names=True, dtype=None,
                      encoding="utf-8"))
    nobj = len(cat_out)
    strid = cat_out[config.id_col]
    args = list(map(str, option['args']))
    mask = np.zeros((nobj, ), dtype=bool)
    if 'all' in args:
        mask = ~mask
        logger.info("plot all objects (total {0})".format(nobj))
        name_suffix = 'all_objs'
    else:
        for o in args:
            for id_ in o.split(','):
                mask[strid == id_] = True
        logger.info("plot objects: {0}".format(', '.join(strid[mask])))
        name_suffix = wrap_if_long('-'.join(strid[mask]), 30)
    # set up canvas
    bulk = cat_out[mask]
    nobj = len(bulk)
    # tile = (1, nobj)
    tile = (nobj, 1)
    width = 250 * tile[1] if option['is_save'] else 1024
    aspect = 0.618 * tile[0] / tile[1]
    strid = bulk[config.id_col]
    print("plotting {0} objects".format(nobj))
    canvas = mympl.CanvasN(
        width=width,
        aspect=aspect,
        scale=1,
        usetw=False,
        share='',
        ngrid=nobj,
        tile=tile,
        )
    fig, axes = canvas.parts()
    fig.set_tight_layout(True)
    ax, bxes = axes[0], axes[1:]
    ax.set_xlabel(tex.tt(r'Wavelength ({\mu}m)'), labelpad=20)
    ax.set_ylabel(tex.tt(r'Flux density (mJy)'), labelpad=40)
    for j, _ in enumerate(bulk[config.id_col]):
        # i = bulk[bulk['strid'] == strid][0]
        plot_fit(bxes[j], bulk[j], config)
    name = 'fig_{0}.eps'.format(name_suffix)
    canvas.save_or_show(name,
                        bbox_inches='tight',
                        # pad_inches=0,
                        )


if __name__ == "__main__":
    pass
