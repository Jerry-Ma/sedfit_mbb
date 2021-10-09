#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-01-20 08:44
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
mbb.py
"""


from __future__ import division
import numpy as np
from lmfit import Model
# from lmfit import conf_interval2d
# from lmfit import conf_interval
# from lmfit import printfuncs
# from lmfit.minimizer import wrap_ueval
from scipy.integrate import quad
# import dill
import multiprocess as mp
import logging


# Constants in SI units
si = dict(
    h=6.62607004e-34,     # m2kg/s
    c=299792458,          # m/s
    k=1.38064852e-23,     # m2kg/s2/K
    )
# Astronomical units
au = dict(
    lsol=3.826e26,        # W = J/s
    msol=1.98855e30,      # kg
    pc=3.08567758e16,     # m
    )
# Empirical quantities
em = dict(
    k850=0.15,            # m2/k
    h0=71,                # km/s/Mpc
    om=0.27,              # 1
    )


def update_progress(mesg, perc):
    print("\r{0:8s}: [{1:40s}] {2:.1f}%".format(
        mesg,
        '#' * int(perc * 40),
        perc * 100), end="", flush=True)
    if perc == 1:
        print("")


class MBB(object):

    out_para_key = ['m_z', 'm_zerr', 'm_dl',
                    'm_lir', 'm_lirerr',
                    'm_lfir',
                    'm_t', 'm_terr',
                    'm_r', 'm_rerr', 'm_md', 'm_mderr', 'm_id', 'm_chi2',
                    'm_cov11', 'm_cov12', 'm_cov21', 'm_cov22']
    out_para_type = ['d', 'd', 'd',
                     'd', 'd',
                     'd',
                     'd', 'd',
                     'd', 'd', 'd', 'd', 'i', 'd',
                     'd', 'd', 'd', 'd']

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        # validate kwargs
        validate_dict = [('beta', 1.5), ('lam0', 100.)]
        for key, defval in validate_dict:
            if key not in kwargs.keys():
                self.logger.warning("key {0} not specified, using default {1}"
                                    .format(key, defval))
        beta = kwargs['beta']
        lam0 = kwargs['lam0']
        # self.model = Model(self.mbb, symtable={'getlir': self.get_lir})
        self.model = Model(self.mbb_lir_z,
                           independent_vars=['wl', ],
                           param_names=['lir', 'tmbb', 'beta', 'lam0', 'z'],
                           )
        # self.model.set_param_hint('reff', value = 1.0)
        self.model.set_param_hint('lir', value=12.0)
        self.model.set_param_hint('tmbb', value=30.,
                                  min=3.,
                                  vary=True)
        self.model.set_param_hint('beta', value=1.5 if beta is None else beta,
                                  min=0., max=3.,
                                  vary=beta is None)
        self.model.set_param_hint('lam0', value=lam0, vary=False)
        self.model.set_param_hint('z', value=1., min=1e-4, max=5., vary=True)
        # self.model.set_param_hint('lir',
        #                           expr='getlir(reff, tmbb, beta, lam0)')
        self.model.set_param_hint(
                'reff',
                expr='10 ** (0.5 * (lir - getlir(1, tmbb, beta, lam0)))')
        self.init_pars = self.model.make_params()
        self.init_pars._asteval.symtable.update(getlir=self.get_lir)

        sed_validator = kwargs.get('validator', None)
        self.sed_validator = self._default_sed_validator \
            if sed_validator is None else sed_validator

    @staticmethod
    def _default_sed_validator(sed, err):
        '''Return whether the input SED is legible and if so
        the valid photometry data mask'''
        good = (sed >= err) & (err > 0)
        return len(sed[good]) >= 3, good

    def batch_fit(self, wls, ids, seds, errs, zs, ts=None, verbose=True):
        '''Perform MBB fitting to a set of SEDs'''
        result = []
        nband, nobj = len(wls), len(ids)
        wls = np.array(np.asarray(wls).view(dtype='d'))
        seds = np.array(seds.tolist(), dtype='d').reshape(nobj, nband)
        errs = np.array(errs.tolist(), dtype='d').reshape(nobj, nband)
        # errs = np.asarray(errs).view(dtype='d').reshape(nobj, nband)
        if ts is not None:
            # ts = np.asarray(ts).view(dtype='d')
            ts = np.array(ts.tolist(), dtype='d')
        pool = mp.Pool(8)
        result = []
        # chunksize = int(nobj / 32) + 1
        # print(seds, errs, zs, ts)
        for i, ret in enumerate(map(
            lambda i: self.fit(wls,
                               seds[i],
                               errs[i],
                               zs[i],
                               t=None if ts is None else ts[i],
                               ind=i,
                               verbose=verbose)[0], range(nobj))):
            update_progress('MBB fit', float(i) / nobj)
            result.append(ret)
        pool.close()
        pool.join()
        update_progress('MBB fit', 1)
        # for i in range(0, nobj):
        #     self.logger.info("working on obj %d", i + 1)
        #     result.append(self.fit(wls,
        #                            seds[i],
        #                            errs[i],
        #                            zs[i],
        #                            t=None if ts is None else ts[i],
        #                            ind=i,
        #                            verbose=verbose)[0])
        return np.hstack(result)

    @staticmethod
    def _unify_array(arr):
        if hasattr(arr, "colnames"):  # astropy table
            return np.array(arr).astype('d').reshape((len(arr.colnames)))
        elif isinstance(arr, np.recarray):
            return arr.astype('d').reshape((len(arr.colnames)))
        elif hasattr(arr, '__array__'):
            return arr
        else:
            return np.array(arr)

    def fit(self, wl, sed, err, z, t=None, ind=0, verbose=True):
        '''
        Perform MBB fitting to given object
            wl (um): input wavelengths
            sed (mJy): fluxes at input wavelengths
            err (mJy): errors of fluxes
        '''
        wl = self._unify_array(wl)
        sed = self._unify_array(sed)
        err = self._unify_array(err)
        # print(wl, sed, err, z, t, ind)
        valid, good = self.sed_validator(sed, err)
        out_para_val = tuple([-1] * len(self.out_para_key))
        if not valid:
            self.logger.warning("object skipped due to"
                                " insufficient number of bands ({0})"
                                .format(len(good)))
            fitresult = None
        else:
            self.init_pars['z'].set(vary=z < 0, value=1. if z < 0 else z,
                                    min=1e-4, max=6)
            if t is None:
                t = -1
            self.init_pars['tmbb'].set(vary=t < 0, value=30. if t < 0 else t,
                                       min=3)
            try:
                fitresult = self.model.fit(sed[good],
                                           wl=wl[good], weights=1. / err[good],
                                           params=self.init_pars)
            except AssertionError as e:
                self.logger.error("minimization failed"
                                  " for object # %d with %s", ind, e)
                fitresult = None
            if fitresult is None:
                pass
            elif not fitresult.success:
                fitresult = None
            # elif fitresult.redchi > 100 or not fitresult.success:  # hard-cut
            #     self.logger.warning("fitting failed with too large chi2 {0}"
            #                         .format(fitresult.redchi))
            #     fitresult = None
            else:
                if fitresult.redchi > 100 or not fitresult.success:
                    self.logger.warning("fitting failed with too"
                                        " large chi2 {0}"
                                        .format(fitresult.redchi))
                best_params = fitresult.params
                bv = best_params.valuesdict()
                be = dict([(p, best_params[p].stderr) for p in bv.keys()])
                # bc = dict([(p, best_params[p].correl) for p in bv.keys()])
                if verbose:
                    self.logger.info(fitresult.fit_report())
                chi2 = fitresult.redchi
                md, mderr = -1, -1
                z = bv['z']
                dl = self.properdist(z) * (1 + z)
                lfir = self.get_lir(
                        bv['reff'],
                        bv['tmbb'],
                        bv['beta'],
                        bv['lam0'],
                        wllo=60, wlup=1000)
                # print(fitresult.covar)
                if fitresult.covar is None:
                    covar = np.ones((2, 2)) * -1
                else:
                    # print(fitresult.var_names)
                    # print(fitresult.nvarys)
                    if fitresult.nvarys == 2:
                        covar = fitresult.covar
                    else:
                        covar = np.ones((2, 2)) * fitresult.covar[0, 0]
                    # covar = fitresult.covar
                out_para_val = (z, be['z'], dl,
                                bv['lir'], be['lir'],
                                lfir,
                                bv['tmbb'], be['tmbb'],
                                bv['reff'], be['reff'], md, mderr, ind, chi2,
                                covar[0, 0], covar[0, 1],
                                covar[1, 0], covar[1, 1],
                                )
        return np.array(
                [out_para_val],
                dtype=list(zip(self.out_para_key, self.out_para_type))), fitresult

    @staticmethod
    def properdist(z):
        c = si['c'] * 1e-3  # in km/s

        def func(z):
            return 1. / np.sqrt(em['om'] * (1 + z) ** 3 + 1 - em['om'])
        dc = quad(func, 0., z)[0] * c / em['h0']  # in Mpc 1e13= km/s -> A/s
        return dc * 1e6 * au['pc']  # in m

    @classmethod
    def mbb_lir_z(cls, wl, lir, tmbb, beta, lam0, z):
        '''
        mbb model with z as parameter
        '''
        dl = cls.properdist(z) * (1 + z)
        restwl = wl / (1 + z)  # um
        restflux = cls.mbb_lir(restwl, lir, tmbb, beta, lam0)  # J/s/Hz
        flux = restflux * 1e3 * 1e26 / 4. / np.pi / dl ** 2 * (1 + z)  # mJy
        return flux

    @classmethod
    def mbb_lir(cls, wl, lir, tmbb, beta, lam0):
        '''
        Single temperature MBB model (Casey 2012):
            Model parameters
                lir (Log Lsun): IR luminosity integrated from 8 to 1000um
                tmbb (K)      : model temperature [K]
                beta          : emissivity
                lam0 (um)     : wavelength where the optical depth is unity
            Data parameters
                wl (um)       : wavelength(s) of interest
            Return:
                snu (W/m^2/Hz): the flux at the input wavelength(s)
        '''
        modified = np.expm1(-(lam0 / wl) ** beta) / np.expm1(-1)
        siwl = wl * 1e-6  # m
        # pi = \int d\Omega
        bb = 2. * np.pi * si['h'] * si['c'] / siwl ** 3 / \
            np.expm1(si['c'] * si['h'] / si['k'] / siwl / tmbb)
        lir_r1 = cls.get_lir(1, tmbb, beta, lam0)
        area = 4 * np.pi * 10 ** (lir - lir_r1) * (1e3 * au['pc']) ** 2
        return area * modified * bb  # J/s/Hz

    @staticmethod
    def mbb(wl, reff, tmbb, beta, lam0):
        '''
        Single temperature MBB model (Casey 2012):
            Model parameters
                reff (kpc)    : emission region
                *lir (Log Lsun): IR luminosity integrated from 8 to 1000um
                tmbb (K)      : model temperature [K]
                beta          : emissivity
                lam0 (um)     : wavelength where the optical depth is unity
            Data parameters
                wl (um)       : wavelength(s) of interest
            Return:
                snu (W/m^2/Hz): the flux at the input wavelength(s)
        '''
        modified = np.expm1(-(lam0 / wl) ** beta) / np.expm1(-1)
        siwl = wl * 1e-6  # m
        # pi = \int d\Omega
        bb = 2. * np.pi * si['h'] * si['c'] / siwl ** 3 / \
            np.expm1(si['c'] * si['h'] / si['k'] / siwl / tmbb)
        area = 4 * np.pi * (reff * 1e3 * au['pc']) ** 2
        return area * modified * bb  # J/s/Hz

    @classmethod
    def _mbb_nu(cls, nu, reff, tmbb, beta, lam0):
        '''For computing integral, we need nu (Hz) and convert it to wl (um)'''
        return cls.mbb(si['c'] * 1e6 / nu, reff, tmbb, beta, lam0)

    @classmethod
    def get_lir(cls, reff, tmbb, beta, lam0, wllo=8, wlup=1000):
        '''
        Compute lir in log sol with given mbb params
            wllo: lower bound of the integration
            wlup: upper bound of the integration
        '''
        lower = si['c'] * 1e6 / wlup
        upper = si['c'] * 1e6 / wllo
        result, _ = quad(cls._mbb_nu, lower, upper,
                         args=(reff, tmbb, beta, lam0))
        lir = np.log10(result / au['lsol'])
        return lir


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    # wl = np.array([246.1857000, 346.1612750, 490.7388000])  # in um
    # sed = np.array([175, 156, 100])
    # err = np.array([5, 6, 5])
    # z = 1.081
    wl = np.array([100., 160., 246.1857000, 346.1612750, 490.7388000])  # in um
    sed = np.array([29.44, 73.8, 79.73712, 63.61015, 43.00635])
    err = np.array([1.83, 3.86, 6.27543, 6.62515, 5.22653])
    z = 1.828
    # z = -1
    # lir = 13.1312 +/- 0.4835
    # t = 50.98 +/- 1.226

    # z = -1

    fitter = MBB(beta=1.5, lam0=100, tmbb=58.7)
    bestfit, fitresult = fitter.fit(wl, sed, err, -1)
    print(bestfit)
    # draw the error ellipse
    # ci, trace = conf_interval(fitresult, fitresult, sigmas=[0.68, 0.95],
    #                           trace=True, verbose=True)
    # printfuncs.report_ci(ci)
    # x, y, gr = conf_interval2d(fitresult, fitresult, 'lir', 'tmbb',
    #         nx=30, ny=30)
    # import matplotlib.pyplot as plt
    # sigmas = [0.68, 0.95, 0.97]
    # plt.contour(x,y,gr, sigmas)
    # plt.errorbar(bestfit['mlir'], bestfit['mtmbb'],
    #         xerr=bestfit['mlirerr'], yerr=bestfit['mterr'],
    #         fmt='o', ms=3)
    # plt.show()
