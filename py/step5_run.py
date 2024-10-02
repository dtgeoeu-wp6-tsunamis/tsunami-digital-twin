import os
import sys
import configparser
import json
import numpy as np
import cartopy
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d

from ptf_preload import load_intensity_thresholds

matplotlib.rcParams['axes.labelsize'] = '14'
matplotlib.rcParams['xtick.labelsize'] = '14'
matplotlib.rcParams['ytick.labelsize'] = '14'
matplotlib.rcParams['legend.fontsize'] = '14'

# def plot_hazard_curve():
# 
#     fig = plt.figure(figsize=(20,14))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_facecolor('#cccccc')
#     ax.grid(True, color='#ffffff')
# 
#     for i in [24]:#range(n_pois):
# 
#         mih_percentiles = np.interp(percentiles, hc[i,::-1], thresholds[::-1])
#         print(mih_percentiles)
# 
#         print("POI ", i, pois[i,:])
#         for key in hc_d.item().keys():
#             print(key, hc_d.item()[key][i])
# 
#         print("HC", hc[i,:])
# 
#         ax.plot(thresholds, hc[i,:], label="py".format(i),
#                  alpha=1, color='#1a5fb4',
#                  linestyle="solid", linewidth=3)
# 
# #    for mih_perc in mih_percentiles:
# #        ax.axvline(x=mih_perc, color="#ff0000")
# 
# #    for perc in percentiles:
# #        ax.axhline(y=perc, color="#0000ff", label=str(perc*100))
# 
# 
#     #ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0., 1.0), frameon=False)
#     # ax.legend(loc="upper right", ncol=1, bbox_to_anchor=(1.0, 1.0), frameon=True)
#     # ax.set_xscale("log")
#     ax.set_yscale("log")
#     # ax.set_xlim(1e-2, 100)
#     ax.set_xlim(0, 20)
#     ax.set_ylim(1e-5, 5)
#     ax.set_xlabel(r'MIH (m)')
#     ax.set_ylabel(r'PoE (50 yrs)')
# 
#     plt.savefig(os.path.join(workdir, "hazard_curve_poi.png"), 
#                 format="png", dpi=150, bbox_inches="tight")
#     plt.close()


def plot_hazard_maps(points, hmaps, event_dict, map_label, fdir, fname):
    """
    """

    proj = cartopy.crs.PlateCarree()
    cmap = plt.cm.magma_r
    ev_lon = event_dict['lon']
    ev_lat = event_dict['lat']
    mmax = [np.amax(v) for k, v in hmaps.items()]
    map_max = np.floor(np.amax(mmax))
    mmin = [np.amin(v) for k, v in hmaps.items()]
    map_min = np.amin(mmin)
    
    for key, hmap in hmaps.items():

        print("mapping ... ", key)
        fig = plt.figure(figsize=(16, 8))
        ax = plt.axes(projection=cartopy.crs.Mercator())
        coastline = cartopy.feature.GSHHSFeature(scale='low', levels=[1])
        #coastline = cartopy.feature.GSHHSFeature(scale='high', levels=[1])
        ax.add_feature(coastline, edgecolor='#000000', facecolor='#cccccc', linewidth=1)
        ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
        ax.add_feature(cartopy.feature.STATES.with_scale('50m'))
        ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'))
        gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1,
                           color="#ffffff", alpha=0.5, linestyle='-')
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

        sc = ax.scatter(points[:,0], points[:,1], c=hmap, s=17, marker="o", 
                    linewidths=0.75, edgecolors="#000000", label=" ",
                    cmap=cmap, clip_on=True, vmin=map_min, vmax=map_max, 
                    transform=proj, zorder=10)

        ax.plot(ev_lon, ev_lat, linewidth=0, marker='*', markersize=14, 
                markerfacecolor='#c0bfbc', markeredgecolor='#000000', 
                transform=proj)

        cbar = plt.colorbar(sc, shrink=0.75)
        #cbar.ax.set_yticklabels(labels=cbar.ax.get_yticklabels(), fontsize=10)
        cbar.set_label(label=f'(m)', size=12)
        ax.set_title("Hazard map - {0}".format(key))
        ax.set_xlabel(r'Longitude ($^\circ$)', fontsize=14)
        ax.set_ylabel(r'Latitude ($^\circ$)', fontsize=14)
        outfile_map = os.path.join(fdir, fname + '_' + key + '_' + map_label + '.png')
        plt.savefig(outfile_map, format='png', dpi=150, bbox_inches='tight')

    
def plot_alert_maps(points, alert_levels, event_dict, points_label, fdir, fname):
    """
    """
    
    proj = cartopy.crs.PlateCarree()
    cmap = matplotlib.colors.ListedColormap(['#00000000', '#39ff00', '#edd400', '#ff0000'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ev_lon = event_dict['lon']
    ev_lat = event_dict['lat']

    for key, alert_level in alert_levels.items():

        print("mapping ... ", key)
        fig = plt.figure(figsize=(16, 8))
        ax = plt.axes(projection=cartopy.crs.Mercator())
        coastline = cartopy.feature.GSHHSFeature(scale='low', levels=[1])
        #coastline = cartopy.feature.GSHHSFeature(scale='high', levels=[1])
        ax.add_feature(coastline, edgecolor='#000000', facecolor='#cccccc', linewidth=1)
        ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
        ax.add_feature(cartopy.feature.STATES.with_scale('50m'))
        ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'))
        gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1,
                           color="#ffffff", alpha=0.5, linestyle='-')
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

        if points_label == "POIs":
            ax.scatter(points[:,0], points[:,1], c=alert_level, s=17, marker="o", 
                        linewidths=0.75, edgecolors="#000000", label=" ",
                        cmap=cmap, clip_on=True, transform=proj, zorder=10, norm=norm)
        elif points_label == "FCPs":
            ax.scatter(points[:,0], points[:,1], c=alert_level, s=40, marker="^", 
                        linewidths=0.75, edgecolors="#000000", label=" ",
                        cmap=cmap, clip_on=True, transform=proj, zorder=10, norm=norm)
        else:
            sys.exit('Error in points_label variable:', points_label)

        ax.plot(ev_lon, ev_lat, linewidth=0, marker='*', markersize=14, 
                markerfacecolor='#c0bfbc', markeredgecolor='#000000', 
                transform=proj)

        res1 = matplotlib.patches.Patch(color='#ff0000',
                                        label=r'Watch',
                                        alpha=1)
        res2 = matplotlib.patches.Patch(color='#edd400',
                                        label=r'Advisory',
                                        alpha=1)
        res3 = matplotlib.patches.Patch(color='#39ff00',
                                        label=r'Information',
                                        alpha=1)

        ax.legend(handles=[res1, res2, res3], ncol=1,
                               borderaxespad=0, frameon=True, 
                               framealpha=0.9)

        ax.set_title("Alert levels at {0} - {1}".format(points_label, key))
        # ax.legend(loc="upper left")
        ax.set_xlabel(r'Longitude ($^\circ$)', fontsize=14)
        ax.set_ylabel(r'Latitude ($^\circ$)', fontsize=14)
        # ax.axis("equal")
        # ax.set_xlim(-5.0, 16.0)
        # ax.set_ylim(32.0, 48.0)
        # ax.set_extent([-8.0, 38.0, 28.0, 48.0], crs=proj)
        # ax.set_extent([lon-5.5, lon+5.5, lat-4.5, lat+4.5], crs=proj)
        outfile_map = os.path.join(fdir, fname + '_' + key + '_' + points_label + '.png')
        plt.savefig(outfile_map, format='png', dpi=150, bbox_inches='tight')

 

def main(**kwargs):

    cfg_file = kwargs.get('cfg_file', None)                       # Configuration file
    workflow_dict = kwargs.get('workflow_dict', None)             # workflow dictionary
    event_dict    = kwargs.get('event_dict', None)                # event dictionary

    Config = configparser.RawConfigParser()
    Config.read(cfg_file)

    fcp_json = Config.get('Files','fcp_json')

    workdir = workflow_dict['workdir']
    percentiles = workflow_dict['percentiles']

    # create plot folder
    figures_dir = os.path.join(workdir, workflow_dict['step5_figures'])
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    # loading intensity thresholds (mih)
    thresholds, intensity_measure = load_intensity_thresholds(cfg = Config)

    pois = np.load(os.path.join(workdir, workflow_dict['pois']), allow_pickle=True).item()
    pois_idx = pois['pois_index']
    pois_coords = pois['pois_coords'][pois_idx]

    fcp_lib = Config.get('Files', 'pois_to_fcp')
    fcp_tmp     = np.load(fcp_lib, allow_pickle=True).item()
    fcp = np.array([fcp_tmp[key][1] for key in fcp_tmp.keys()])

    #hc = np.load(os.path.join(workdir, 'Step3_hazard_curves.npy'))
    hc_sparse = scipy.sparse.load_npz(os.path.join(workdir, workflow_dict['step3_hc'] + '.npz'))
    hc = hc_sparse.toarray()
    n_pois, n_thr = hc.shape

    hmaps = np.load(os.path.join(workdir, workflow_dict['step3_hc_perc'] + '.npy'), allow_pickle=True).item()
    # hmaps_mean = hmaps['mean']
    # print(hc.shape, hmaps.keys(), hmaps_mean.shape)
    
    pois_al = np.load(os.path.join(workdir, workflow_dict['step4_alert_levels_POI']),
                      allow_pickle=True).item()
    fcp_al = np.load(os.path.join(workdir, workflow_dict['step4_alert_levels_FCP']),
                     allow_pickle=True).item()

    plot_alert_maps(pois_coords, pois_al, event_dict, "POIs", 
                    figures_dir, workflow_dict['step5_alert_levels'])
    plot_alert_maps(fcp, fcp_al, event_dict, "FCPs", 
                    figures_dir, workflow_dict['step5_alert_levels'])

    plot_hazard_maps(pois_coords, hmaps, event_dict, "HazMap", 
                     figures_dir, workflow_dict['step5_hazard_maps'])


if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
