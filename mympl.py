#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2015-12-23 20:24
# Python Version :  %PYVER%
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
mympl.py

Provide Jerry's matplotlib style (v1) and relevant utility functions
"""


from __future__ import division
import os
import sys
import logging
from cycler import cycler
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import gridspec
# import subprocess
from math import sqrt, ceil
import numpy as np
# import matplotlib.patches as mpatches

from matplotlib import rc
from functools import reduce

NAME = "MYMPL"
USE_TEX = False


def tex_font_settings(**kwargs):
    settings = {
        'serif': [  # 'Times', 'Palatino', 'New Century Schoolbook', 'Bookman',
                  'Computer Modern Roman'],
        'sans-serif': [  # 'Helvetica', 'Avant Garde',
                       'Computer Modern Sans serif'],
        'monospace': [  # 'Courier',
                      'Computer Modern Typewriter'],
        'weight': 'regular',
        'family': 'serif',
        'size': 12,
        }
    for key, value in kwargs.items():
        if value is not None:
            settings[key] = value
    return settings


emulateapj = '/home/ma/Codes/writing/HerschelQuasar/herschelquasar.width'
LATEX_INCHES_PER_PT = 1.0 / 72.27


class CanvasBase(object):
    '''Base class for providing quick layout/sizing'''

    inches_per_pt = LATEX_INCHES_PER_PT

    def __init__(self, width=None, aspect=0.618, scale=1, usetw=False,
                 fontsize=12, family='serif', projection=None):
        '''
        width:
            parse a latex size configure file, set the width of figure
            accordingly
        demr:
            the aspect ratio of the plot
        scale:
            scale to apply to the set dem parameter
        usetw:
            use text width as the figure width if True, otherwise colwidth
        '''
        self.logger = logging.getLogger(NAME)
        self.init_mplrc(width, aspect, scale, usetw, fontsize, family)
        self.fig = plt.figure()
        self.axes = []
        self.ax_keys = {}
        if projection is not None:
            self.ax_keys['projection'] = projection

    def parts(self):
        return self.fig, self.axes

    def init_mplrc(self, width, aspect, scale, usetw, fontsize, family):

        if isinstance(width, str):
            with open(width) as fo:
                col_width_pt, font_size_pt, text_width_pt = [
                    float(ln.split('=')[-1].strip().rstrip('pt'))
                    for ln in fo.readlines()[:3]]
        elif isinstance(width, float) or isinstance(width, int):
            col_width_pt = width
            font_size_pt = fontsize
            text_width_pt = col_width_pt * 2
        elif width is None:
            col_width_pt = 8.0 / self.inches_per_pt
            font_size_pt = fontsize
            text_width_pt = col_width_pt * 2
        else:
            raise ValueError('width parameter cannot be recognised')
        fig_width_pt = text_width_pt if usetw else col_width_pt
        fig_width = fig_width_pt * self.inches_per_pt * scale
        fig_height = fig_width * aspect
        self.logger.info(
            'figure size: {0:.1f}in by {1:.1f}in ({2:.0f}x{3:.0f})'.format(
                fig_width, fig_height, fig_width_pt, fig_width_pt * aspect))
        if USE_TEX:
            rc('font', **tex_font_settings(family=family, size=font_size_pt))
        else:
            rc('font', family=family, size=font_size_pt)
        rc('legend', fontsize=font_size_pt)
        rc('figure',
           figsize=(fig_width, fig_height), dpi=1 / self.inches_per_pt)
        self.font_size_pt = font_size_pt

    def save_or_show(self, savename,
                     bbox_inches='tight',
                     pad_inches='0.2em',
                     save=None,
                     **kwargs):
        '''Save figure with given name or show the plot, depending on
        the last sys.argv'''
        if save is None:
            argv = sys.argv[1:]
            if not argv:
                save = False
            else:
                try:
                    s = int(argv[-1])
                    save = True if s == 1 else False
                except ValueError:
                    if argv[-1].lower() in ['true', 'save']:
                        save = True
                    else:
                        save = False
        if isinstance(pad_inches, str) and pad_inches[-2:].lower() == 'em':
            pad_inches = float(pad_inches[:-2]) * \
                    self.font_size_pt * self.inches_per_pt
        else:
            pad_inches = float(pad_inches)
        if save:
            self.fig.savefig(
                savename,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                dpi='figure',
                format=os.path.splitext(savename)[-1][1:], **kwargs)
            self.logger.info('figure saved: {0}'.format(savename))
        else:
            plt.show()

    @staticmethod
    def let_dummy(ax, tick=True):
        ax.set_frame_on(False)
        if tick:
            ax.tick_params(labelcolor=(1, 1, 1, 0),
                           top='off', bottom='off',
                           left='off', right='off')
        else:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
        # ax.spines['top'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['right'].set_color('none')
        ax.patch.set_visible(False)


class CanvasOne(CanvasBase):

    def __init__(self, **kwargs):
        super(CanvasOne, self).__init__(**kwargs)
        ax = self.fig.add_subplot(1, 1, 1, **self.ax_keys)
        self.axes.append(ax)


class CanvasTwo(CanvasOne):
    def __init__(self, ratio=[1, 1], direction='h', share='auto', space=None,
                 **kwargs):
        super(CanvasTwo, self).__init__(**kwargs)
        # let it be the placeholder for shared axis labels
        self.let_dummy(self.axes[0], tick=False)
        if direction == 'h':
            gs = gridspec.GridSpec(1, 2, width_ratios=ratio)
            if share == 'auto':
                share = 'y'
        elif direction == 'v':
            gs = gridspec.GridSpec(2, 1, height_ratios=ratio)
            if share == 'auto':
                share = 'x'
        else:
            raise ValueError("unknown direction '{0}'".format(direction))
        bx = self.fig.add_subplot(gs[0])
        if share == 'y':
            cx = self.fig.add_subplot(gs[1], sharey=bx)
            # cx.yaxis.set_major_formatter(plt.NullFormatter())
            for ylabel in cx.get_yticklabels():
                ylabel.set_visible(False)
        elif share == 'x':
            cx = self.fig.add_subplot(gs[1], sharex=bx)
            # cx.xaxis.set_major_formatter(plt.NullFormatter())
            for xlabel in cx.get_xticklabels():
                xlabel.set_visible(False)
        elif share == 'n':
            cx = self.fig.add_subplot(gs[1])
        else:
            raise ValueError(
                    "unknown key '{0}' for argument share".format(share))
        if space is not None:
            if direction == 'h':
                self.fig.subplots_adjust(wspace=space)
            elif direction == 'v':
                self.fig.subplots_adjust(hspace=space)
        self.axes.extend([bx, cx])


class CanvasN(CanvasBase):
    def __init__(self, ngrid=1, tile=None, share='xy', hide_inner_tick='',
                 wspace=0.02, hspace=0.02, hide_last_tick=False, **kwargs):
        super(CanvasN, self).__init__(**kwargs)
        ax = self.fig.add_subplot(1, 1, 1)
        self.axes.append(ax)
        self.let_dummy(self.axes[0])
        if tile is None:
            ncol = int(sqrt(ngrid) / kwargs['aspect'])
            nrow = int(ceil(ngrid / ncol))
        else:
            nrow, ncol = tile
        self.logger.info('figure grid: {0} {1}x{2}'.format(ngrid, nrow, ncol))
        xtickvis = [False] * (ncol * (nrow - 1)) + \
                   [True] * (ngrid - ncol * (nrow - 1))
        ytickvis = ([True] + [False] * (ncol - 1)) * nrow
        xtickadj = [False] * (ncol * (nrow - 1)) + \
                   [True] * (ncol - 1) + [False]
        ytickadj = [False] * ncol + \
            ([True] + [False] * (ncol - 1)) * (nrow - 1)
        xshare = [None] * ngrid
        yshare = [None] * ngrid
        if 'x' in share:
            xshare = [None] * ncol + range(0, ncol) * (nrow - 1)
        if 'y' in share:
            yshare = reduce(lambda a, b: a + b,
                            [[None] + [i * ncol] * (ncol - 1)
                             for i in range(0, nrow)])
        zorder = reduce(lambda a, b: a + b,
                        [[i] * ncol for i in range(nrow, 0, -1)])
        self.fig.subplots_adjust(left=0.05, right=0.97, bottom=0.08, top=0.97,
                                 wspace=wspace, hspace=hspace)
        bxes = []
        for i in range(0, ngrid):
            if xshare[i] is not None:
                xshare[i] = bxes[xshare[i]]
            if yshare[i] is not None:
                yshare[i] = bxes[yshare[i]]
            ax_keys = {k: v[i] for k, v in self.ax_keys.items()}
            bxes.append(
                self.fig.add_subplot(nrow, ncol, i + 1,
                                     sharex=xshare[i],
                                     sharey=yshare[i],
                                     zorder=zorder[i],
                                     **ax_keys))
            # remove the first and last label if share
            if 'x' in share or 'x' in hide_inner_tick:
                matplotlib.artist.setp(
                    bxes[-1].get_xticklabels(), visible=xtickvis[i])
                if hide_last_tick and xtickadj[i]:
                    xticks = bxes[-1].xaxis.get_major_ticks()
                    xticks[-1].label1.set_visible(False)
            if 'y' in share or 'y' in hide_inner_tick:
                matplotlib.artist.setp(
                    bxes[-1].get_yticklabels(), visible=ytickvis[i])
                if hide_last_tick and ytickadj[i]:
                    yticks = bxes[-1].yaxis.get_major_ticks()
                    yticks[-1].label1.set_visible(False)
        self.axes.extend(bxes)
        # for custom use later on
        self.xtickvis = xtickvis
        self.ytickvis = ytickvis
        self.xtickadj = xtickadj
        self.ytickadj = ytickadj
        self.xtickadj = xshare
        self.ytickadj = yshare


class HCColor(object):
    kelly = np.array([
        [255, 179, 0], [128, 62, 117], [255, 104, 0], [166, 189, 215],
        [193, 0, 32], [206, 162, 98], [129, 112, 102], [0, 125, 52],
        [246, 118, 142], [0, 83, 138], [255, 122, 92], [83, 55, 122],
        [255, 142, 0], [179, 40, 81], [244, 200, 0], [127, 24, 13],
        [147, 170, 0], [89, 51, 21], [241, 58, 19], [35, 44, 22]]) / 255.
    paul = np.array([
        [240, 163, 255], [0, 117, 220], [153, 63, 0], [76, 0, 92],
        [25, 25, 25], [0, 92, 49], [43, 206, 72], [255, 204, 153],
        [128, 128, 128], [148, 255, 181], [143, 124, 0], [157, 204, 0],
        [194, 0, 136], [0, 51, 128], [255, 164, 5], [255, 168, 187],
        [66, 102, 0], [255, 0, 16], [94, 241, 242], [0, 153, 143],
        [224, 255, 102], [116, 10, 255], [153, 0, 0], [255, 255, 128],
        [255, 255, 0], [255, 80, 5]]) / 255.


class SolarizedColor(object):

    base03 = "#002b36"
    base02 = "#073642"
    base01 = "#586e75"
    base00 = "#657b83"
    base0 = "#839496"
    base1 = "#93a1a1"
    base2 = "#eee8d5"
    base3 = "#fdf6e3"
    yellow = "#b58900"
    orange = "#cb4b16"
    red = "#dc322f"
    magenta = "#d33682"
    violet = "#6c71c4"
    blue = "#268bd2"
    cyan = "#2aa198"
    green = "#859900"
    colortable = [red, blue, yellow, magenta, orange, green, cyan, violet]

    @staticmethod
    def hsv(c, h=None, s=None, v=None, frac=False):
        '''Quickly change the color in hsv space'''
        if isinstance(c, str):
            rgb = np.array([[mc.hex2color(c), ]])
        else:  # rgb
            rgb = np.array([[c[:3], ]])
        hsv = mc.rgb_to_hsv(rgb)
        # print c
        # print rgb
        # print hsv
        for i, j in enumerate([h, s, v]):
            if j is not None:
                if frac:
                    if j < 1:
                        hsv[0][0][i] = hsv[0][0][i] * j
                    else:
                        hsv[0][0][i] = hsv[0][0][i] + \
                                (1 - hsv[0][0][i]) * (1 - j)
                else:
                    hsv[0][0][i] = j
        return mc.hsv_to_rgb(hsv)[0][0]

    @classmethod
    def wash(cls, c):
        return cls.hsv(c, s=0.1, v=0.9)

    @staticmethod
    def blend(c1, c2, a=0.5):
        if isinstance(c1, str):
            c1 = mc.hex2color(c1)
        if isinstance(c2, str):
            c2 = mc.hex2color(c2)
        return (c1[0] * a + (1 - a) * c2[0],
                c1[1] * a + (1 - a) * c2[1],
                c1[2] * a + (1 - a) * c2[2],
                )


class TexStyle(object):
    '''provide utility to style the text using TEX.'''

    @classmethod
    def wrap(cls, s, st):
        if isinstance(s, str):
            ln = s.split('\n')
            return '\n'.join([r'$\%s{%s}$' % (st, cls._escape(c)) for c in ln])
        else:
            return map(lambda ss: cls.wrap(ss, st), s)

    @classmethod
    def rm(cls, s):
        return cls.wrap(s, 'mathrm')

    @classmethod
    def tt(cls, s):
        return cls.wrap(s, 'mathtt')

    @classmethod
    def sf(cls, s):
        return cls.wrap(s, 'mathsf')

    @classmethod
    def noindent(cls, s):
        return r'\noindent ' + s

    @staticmethod
    def _escape(s):
        _s = s.replace('_{', '`"')
        _s = _s.replace('_', '-').replace('-', r'\text{--}')
        return _s.replace(' ', r'\ ').replace('`"', '_{')


class MplStyle(object):

    '''provide utility to style the text using native approach.'''

    @classmethod
    def wrap(cls, s, st):
        if isinstance(s, str):
            ln = s.split('\n')
            return '\n'.join([r'$\%s{%s}$' % (st, cls._escape(c)) for c in ln])
        else:
            return map(lambda ss: cls.wrap(ss, st), s)

    @classmethod
    def rm(cls, s):
        return cls.wrap(s, 'mathrm')

    @classmethod
    def tt(cls, s):
        return cls.wrap(s, 'mathtt')

    @classmethod
    def sf(cls, s):
        return cls.wrap(s, 'mathsf')

    @classmethod
    def noindent(cls, s):
        return s

    @staticmethod
    def _escape(s):
        _s = s.replace('_{', '`"')
        _s = _s.replace('_', '-').replace('-', r'{-}')
        return _s.replace(' ', r'\ ').replace('`"', '_{')


if USE_TEX:
    TextStyle = TexStyle
else:
    TextStyle = MplStyle

# global rc params
matplotlib.style.use('default')
rc('ps', usedistiller='xpdf')
if USE_TEX:
    rc('font', **tex_font_settings())
    rc('text', **{
        'usetex': True,
        'latex.unicode': True,
        'latex.preamble': [r'\usepackage{amsmath}', ],
        })
rc('legend', **{'numpoints': 1,
                'scatterpoints': 1,
                # 'handletextpad': 0,
                })
rc('axes', prop_cycle=cycler('color', SolarizedColor.colortable))


def use_hc_color(key):
    rc('axes', prop_cycle=cycler('color', getattr(HCColor, key)))


def get_dummy_leg():
    return matplotlib.patches.Rectangle((1, 1), 1, 1,
                                        fill=False, edgecolor='none',
                                        visible=False), \
            dict(handletextpad=0, handlelength=0, frameon=False)


logging.basicConfig(format='[%(name)s] %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    SolarizedColor.wash(SolarizedColor.red)
