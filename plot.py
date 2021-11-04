#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-01-10 15:00
# Python Version :  2.7.10
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
plot_fit.py
"""

from __future__ import division
import numpy as np

from .mympl import SolarizedColor as sc
from .mympl import TextStyle as ts
import matplotlib
from .mbb import MBB


def plot_fit(ax, entry, config):

    fitwl = np.array(config.sed_wls).astype(dtype='d')
    fitsed = np.array(
            [entry[i] for i in config.sed_flux_cols]).astype(dtype='d')
    fiterr = np.array(
            [entry[i] for i in config.sed_err_cols]).astype(dtype='d')
    good = (fitsed > fiterr) & (fiterr > 0)
    fitsed = fitsed[good]
    fiterr = fiterr[good]
    fitwl = fitwl[good]
    if len(fitwl) == 0:
        print("Not enough good point")
        return
    z = entry[config.z_col]
    name = entry[config.id_col]
    if config.t_col not in entry.dtype.names:
        t = -1
    else:
        t = entry[config.t_col]
    prop = ['m_chi2', ('m_lir', 'm_lirerr'), 'm_lfir']
    if config.leg_name:
        lbl_sed = '{0:s}\n'.format(name)
    else:
        lbl_sed = ''
    if z < 0:
        prop.append(('m_z', 'm_zerr'))
    else:
        lbl_sed += '\n z={0:s}'.format(
                'N/A' if z < 0 else config.leg_z_fmt.format(z)).strip()
    if t < 0:
        prop.append(('m_t', 'm_terr'))
    else:
        lbl_sed += '\nT_{{mbb}}={0:s} K'.format(
            'N/A' if t < 0 else config.leg_t_fmt.format(t))
    for mdl_wl, mdl_flux_col in zip(config.mdl_wls, config.mdl_flux_cols):
        lbl_sed += '\nS_{{{:.0f}um}}={:.2f} mJy'.format(
            float(mdl_wl), entry[f'm_{mdl_flux_col}']
            )
    print(lbl_sed)
    lbl_sed = ts.tt(lbl_sed)

    pltkw_line = dict(lw=1.2, linestyle='-')
    pltkw_fit = dict(pltkw_line, color=sc.red)
    pltkw_fitsed = dict(fmt='o', ms=12, fillstyle='none',
                        color=sc.green, mec=sc.green,
                        elinewidth=1.0,
                        mew=1.0,
                        zorder=3)
    pltkw_mdlsed = dict(marker='o', ms=6, fillstyle='none',
                        color=sc.violet, mec=sc.violet,
                        linestyle='none',
                        zorder=2)
    legkw_temp = dict(loc='lower center',
                      ncol=1,
                      handlelength=1.5,
                      handletextpad=0.5)
    legkw_sed = dict(legkw_temp, loc=config.leg_loc,
                     frameon=False,
                     handletextpad=0,
                     handlelength=0.5)
    ax.set_xscale('log')

    mdl_wls, mdl_flux_cols = config.get_mdl_flux_info()

    xlim_min = np.min(mdl_wls) * 0.5
    xlim_max = np.max(mdl_wls) * 2.0
    ax.set_xlim((xlim_min, xlim_max))
    # ax.set_xlim((60, 1000))
    ax.set_yscale('log')
    ylim_min = np.min(fitsed) * 0.5
    ylim_max = np.max(fitsed) * 2.0
    ax.set_ylim((ylim_min, ylim_max))
    # ax.set_ylim((5, 500))

    ax.errorbar(fitwl, fitsed, yerr=fiterr, **pltkw_fitsed)
    if entry['m_chi2'] > 0:
        # get template by call the MBB class method
        temp_wl = np.logspace(np.log10(xlim_min), 4, 500)
        temp_flux = MBB.mbb_lir_z(
                temp_wl, entry['m_lir'], entry['m_t'],
                config.beta, config.lambda0, entry['m_z'])
        leg_fit, = ax.plot(temp_wl, temp_flux, **pltkw_fit)
        lbl_fit = tex_lbl(entry, prop, 'MBB')
        leg = ax.legend([leg_fit], [lbl_fit], **legkw_temp)
        # fix handle alignment
        cb = leg._legend_box._children[-1]._children[0]
        for ib in cb._children:
            ib.align = "center"
        leg.legendPatch.set_alpha(0)
        # mdl flux
        if config.plot_mdl_flux:
            ax.plot(
                mdl_wls,
                [entry[c] for c in mdl_flux_cols], **pltkw_mdlsed)

    else:
        leg = None
    dummy = matplotlib.patches.Rectangle((1, 1), 1, 1, fill=False,
                                         edgecolor='none',
                                         visible=False)
    ax.legend([dummy, ], [lbl_sed, ], **legkw_sed)
    if leg is not None:
        ax.add_artist(leg)


def tex_lbl(entry, prop, title):
    # latex-ize the strings
    texized_key = {
                   'm_chi2': r'\chi^2',
                   'm_lir': r'Log L_{IR}/L_{\odot}',
                   'm_lfir': r'Log L_{FIR}/L_{\odot}',
                   'm_t': r'T',
                   'm_z': r'z',
                   # 'reff': r'R_{eff}',
                   # 'sfrd2d': r'\Sigma_{SFR}',
                   }
    texized_fmt = {
                   'm_chi2': r'{0:.1f}',
                   'm_lir': r'{0:.1f}_{{-{1:.1f}}}^{{+{1:.1f}}}',
                   'm_lfir': r'{0:.1f}',
                   'm_t': r'{0:.1f}_{{-{1:.1f}}}^{{+{1:.1f}}} K',
                   'm_z': r'{0:.2f}_{{-{1:.2f}}}^{{+{1:.2f}}}',
                   }
    s = []
    for j, i in enumerate(prop):
        if isinstance(i, str):
            s.append('{0:s}{{=}}{1:s}'.format(
                texized_key[i], texized_fmt[i].format(entry[i])))
        else:
            s.append('{0:s}{{=}}{1:s}'.format(
                texized_key[i[0]], texized_fmt[i[0]].format(
                    entry[i[0]], entry[i[1]])))
    # s = tex_mathtt(r'\\'.join(s)).replace('\n', r'\\').replace(' ', '\ ')
    if title is not None:
        s[0] = r'{0:s}, '.format(title) + s[0]
    s = ts.noindent('\n'.join(ts.tt(s)))
    return s


if __name__ == '__main__':

    pass
    # bulk = Table.read('./bulk.asc', format='ascii.commented_header')
    # tempdump = np.load('./cmctempdump.npz')
    # nobj = len(bulk)
    # canvas = mympl.CanvasN(
    #     width=mympl.emulateapj,
    #     ngrid=nobj,
    #     tile=[4, 4],
    #     aspect=0.618,
    #     scale=1.6,
    #     usetw=True,
    #     share='xy',
    #     )
    # fig, axes = canvas.parts()
    # ax = axes[0]  # dummy axes for lable
    # bxes = axes[1:]
    # ax.set_xlabel(ts.tt(r'Wavelength ({\mu}m)'))
    # ax.set_ylabel(ts.tt(r'Flux density (mJy)'))

    # objects = bulk['strid']
    # for j, strid in enumerate(objects):
    #     i = bulk[bulk['strid'] == strid][0]
    #     plot_fit(bxes[j], i,
    #              tempdump['cmcspire'][i['cmctemp']],
    #             )

    # name = 'fig_lens_fit.eps'
    # canvas.save_or_show(name,
    #                     bbox_inches='tight',
    #                     pad_inches=0.2,
    #                     )
