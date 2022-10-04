#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-07-11 18:00
# Python Version :  2.7.11
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
sedfit_example.py

An example (template) of how to configure and run an SED fitting job
"""

# --------------#-------- Start of Configuration -----------------------------#
# input catalog #
# ------------- #
# filename of input catalog
cat_in = "./catalog.in"
# column name of the object identifier (name)
id_col = 'strid'
# column name of the redshifts
# object with z < 0 will be fitted with z as free parameter
z_col = 'z'
# column name of the temperature
# object with T < 0 will be fitted with T as free parameter
t_col = 't'
# wavelengths [micron] of input SEDs
sed_wls = [100., 160., 246.1857, 346.161275, 490.7388]
# column names of the SED flux [mJy], in the same order as the wavelengths
sed_flux_cols = ['F100', 'F160', 'F250', 'F350', 'F500']
# column names of the SED flux uncertainty [mJy]
sed_err_cols = ['E100', 'E160', 'E250', 'E350', 'E500']
# wavelengths [micron] of additional output model fluxes 
mdl_wls = [850.]
# column names of output model fluxes.
mdl_flux_cols = ['F850']
# ------------- #
#  SED fitting  #
# ------------- #
# wavelength [micron] where the opacity is 1 ("anchor" wavelength).
lambda0 = 100.
# emissivity, 'beta = None' means fit as free parameter
beta = 1.5
# -------------- #
# output catalog #
# -------------- #
# filename of output catalog (a horizontal concat. of input cat and fit result)
cat_out = "./catalog.out"
# npz archive of the best fit templates
temp_out = "./temp_out"  # not implemented
# whether to extract the best fit templates in ascii format
unpack = True  # not implemented
# directory to hold the unpacked best-fit templates
unpack_dir = './temp_unpacked'  # not implemented

# ---------------#
# plot           #
# ---------------#
leg_name = True
leg_z_fmt = "{:.1f}"
leg_t_fmt = "{:.0f}"

# output catalog keys:
# m_z     : redshift
# m_zerr  : uncertainty of redshift
# m_dl    : luminosity distance
# m_lir   : total IR luminosity 8-1000um (Log Lsun)
# m_lirerr: uncertainty of m_lir
# m_t     : model temperature (K)
# m_terr  : uncertainty of m_t
# m_r     : effective radius of the emitting region (kpc)
# m_rerr  : uncertainty of m_r
# m_md    : dust mass (Log Msun)
# m_mderr : uncertainty of m_md
# m_id    : index into the best-fit template archive
# m_chi2  : chi square of the fitting
# m_cov11 : covariance matrix between m_lir and m_tmbb, 11
# m_cov12 : covariance matrix between m_lir and m_tmbb, 12
# m_cov21 : covariance matrix between m_lir and m_tmbb, 21
# m_cov22 : covariance matrix between m_lir and m_tmbb, 22
# ----------------------- End of Configuration -------------------------------#


if __name__ == "__main__":

    import os
    import sys
    # add the directory that contains sedfit source code to your python path
    sys.path.insert(0, os.path.expanduser("~/Codes/pyzma"))
    from sedfit import core
    core.bootstrap()
