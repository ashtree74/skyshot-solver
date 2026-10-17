#!/usr/bin/env python3
"""Regenerate tests/fixtures/panorama-wcs.json using independent WCSLIB transforms.
Run with astropy + numpy; only the generated small fixture is needed by Rust tests.
"""
import json
from pathlib import Path
import astropy
import numpy as np
from astropy.io.fits import Header
from astropy.wcs import WCS


def unit(ra, dec):
    a, d = np.deg2rad([ra, dec])
    return np.array([np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)])


fixtures = []
for proj in ['STG', 'CAR', 'MER', 'CYP']:
    for pole in ([None] if proj == 'STG' else [None, [40., 60.], [140., -15.]]):
        center = [283., -4.]
        a, d = np.deg2rad(center)
        c = unit(*center)
        n = unit(*pole) if pole else np.array([-np.sin(d)*np.cos(a), -np.sin(d)*np.sin(a), np.cos(d)])
        lat0 = np.arcsin(n@c)
        eq = c-(n@c)*n
        eq /= np.linalg.norm(eq)
        east = np.cross(n, eq)
        wcs = dict(ra=center[0], dec=center[1], crpix=[682., 1023.5],
                   cd=[[-.075, .002], [.003, .065]], sip=None, proj=proj)
        if pole:
            wcs['native_pole'] = pole
        h = Header(dict(NAXIS=2, CTYPE1=f'RA---{proj}', CTYPE2=f'DEC--{proj}',
                        CRVAL1=center[0], CRVAL2=center[1], CRPIX1=683., CRPIX2=1024.5,
                        CD1_1=-.075, CD1_2=.002, CD2_1=.003, CD2_2=.065))
        if proj != 'STG':
            h['PV1_0'] = 1
            h['PV1_1'] = 0.
            h['PV1_2'] = np.rad2deg(lat0)
            h['LONPOLE'] = np.rad2deg(np.arctan2(east[2],eq[2]))
            h['LATPOLE'] = np.rad2deg(np.arcsin(n[2]))
            if proj == 'CYP':
                h['PV2_1'], h['PV2_2'] = 0., 1.
        pixels = [[682., 1023.5], [10., 200.], [1300., 1700.], [400., 700.]]
        sky = WCS(h).all_pix2world(pixels, 0).tolist()
        valid = [(p,s) for p,s in zip(pixels,sky) if np.isfinite(s).all()]
        params={k:h[k] for k in ['PV1_0','PV1_1','PV1_2','LONPOLE','LATPOLE','PV2_1','PV2_2'] if k in h}
        fixtures.append(dict(wcs=wcs, fits_parameters=params, pixels=[p for p,_ in valid], sky=[s for _,s in valid]))
Path('tests/fixtures/panorama-wcs.json').write_text(json.dumps(dict(generator='bench/panorama_wcs_fixture.py', astropy=astropy.__version__, fixtures=fixtures), indent=2, allow_nan=False)+'\n')
