import os
import sys
import numpy as np
import cartopy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy
# import configparser
# import json
# from scipy.interpolate import interp1d

# from pyptf.ptf_preload import load_intensity_thresholds

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
#     #ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0., 1.0), frameon=False)7.574120044708252sec
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


def plot_hazard_maps(points, hmaps, event_dict, map_label, logo, fdir, fname, logger):
    """
    """

    proj = cartopy.crs.PlateCarree()
    #cmap = plt.cm.magma_r
    cmap = plt.cm.jet
    ev_lon = event_dict['lon']
    ev_lat = event_dict['lat']
    ev_depth = event_dict['depth']
    ev_mag = event_dict['mag']
    ev_place = event_dict['place']
    # mmax = [np.amax(v) for k, v in hmaps.items()]
    # mmin = [np.amin(v) for k, v in hmaps.items()]
    # map_max = np.floor(np.amax(mmax))
    # map_min = np.amin(mmin)

    for key, hmap in hmaps.items():           
                
        logger.info("mapping ... {}".format(key))
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
                    cmap=cmap, clip_on=True,# vmin=map_min, vmax=map_max, 
                    transform=proj, zorder=10, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=50))#(vmin=map_min, vmax=map_max))

        ax.plot(ev_lon, ev_lat, linewidth=0, marker='*', markersize=14, 
                #markerfacecolor='#c0bfbc', markeredgecolor='#000000', 
                markerfacecolor='magenta', markeredgecolor='#000000', 
                transform=proj)

        cbar = plt.colorbar(sc, shrink=0.75)
        #cbar.ax.set_yticklabels(labels=cbar.ax.get_yticklabels(), fontsize=10)
        cbar.set_label(label=f'(m)', size=12)
        # ax.set_title("Hazard map - {0}".format(key))
        plt.suptitle("Hazard map - {0}".format(key),fontsize=24)
        plt.title("Epicentral Region: {0} \n Event parameters: Lon={1}, Lat={2}, Depth={3}; Magnitude={4}".format(ev_place, ev_lon, ev_lat, ev_depth, str(ev_mag)[:3] ),fontsize=18)
        ax.set_xlabel(r'Longitude ($^\circ$)', fontsize=14)
        ax.set_ylabel(r'Latitude ($^\circ$)', fontsize=14)
        
        arr_img = plt.imread(logo)
        im = OffsetImage(arr_img, zoom=0.1)
        ab = AnnotationBbox(im, (0, 0), xycoords='axes fraction', box_alignment=(-0.12,-0.3))
        ax.add_artist(ab)

        outfile_map = os.path.join(fdir, fname + '_' + key + '_' + map_label + '.png')
        plt.savefig(outfile_map, format='png', dpi=150, bbox_inches='tight')
        plt.close()

    
def plot_alert_maps(pois, al_pois, fcp, al_fcp, event_dict, logo, fdir, fname, logger):
    """
    """
    
    proj = cartopy.crs.PlateCarree()
    cmap = matplotlib.colors.ListedColormap(['#00000000', '#39ff00', '#edd400', '#ff0000'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ev_lon = event_dict['lon']
    ev_lat = event_dict['lat']
    ev_depth = event_dict['depth']
    ev_mag = event_dict['mag']
    ev_place = event_dict['place']

    res1 = matplotlib.patches.Patch(color='#ff0000',
                                    label=r'Watch',
                                    alpha=1)
    res2 = matplotlib.patches.Patch(color='#edd400',
                                    label=r'Advisory',
                                    alpha=1)
    res3 = matplotlib.patches.Patch(color='#39ff00',
                                    label=r'Information',
                                    alpha=1)

    # for key, alert_level in al_pois.items():
    for keys in zip(al_pois, al_fcp):

        key = keys[0]
        al1 = al_pois[key]
        al2 = al_fcp[key]

        arr_img = plt.imread(logo)
        im = OffsetImage(arr_img, zoom=0.1)

        logger.info("mapping ... {}".format(key))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16), subplot_kw={'projection': cartopy.crs.Mercator()})
        fig.set_tight_layout(True)
        # plt.suptitle("Event parameters: Lon={1}, Lat={2}, Depth={3}; Magnitude={4} - Epicentral Region: {0} \n".format(ev_place, ev_lon, ev_lat, ev_depth, str(ev_mag)[:3] ),fontsize=14)
        text = "Event parameters: Lon={1}, Lat={2}, Depth={3}; Magnitude={4} - Epicentral Region: {0}".format(ev_place, ev_lon, ev_lat, ev_depth, str(ev_mag)[:4])
        plt.text(0.5, -0.1, text, horizontalalignment='center',verticalalignment='center', transform=ax2.transAxes, fontsize=16, bbox={'facecolor': '#dddddd', 'alpha': 0.5, 'pad': 5})
        
        # ax1: plot Al at POIs  
        # ax1 = plt.axes(projection=cartopy.crs.Mercator())
        coastline = cartopy.feature.GSHHSFeature(scale='low', levels=[1])
        ax1.add_feature(coastline, edgecolor='#000000', facecolor='#cccccc', linewidth=1)
        ax1.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
        ax1.add_feature(cartopy.feature.STATES.with_scale('50m'))
        ax1.add_feature(cartopy.feature.OCEAN.with_scale('50m'))
        
        gl = ax1.gridlines(crs=proj, draw_labels=True, linewidth=1,
                            color="#ffffff", alpha=0.5, linestyle='-')
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

        ax1.plot(fcp[:,0], fcp[:,1], linewidth=0, clip_on=True, transform=proj, zorder=10)
        ax1.scatter(pois[:,0], pois[:,1], c=al1, s=17, marker="o", 
                    linewidths=0.75, edgecolors="#000000", label=" ",
                    cmap=cmap, clip_on=True, transform=proj, zorder=10, norm=norm)
        
        ax1.plot(ev_lon, ev_lat, linewidth=0, marker='*', markersize=14, 
                 markerfacecolor='white', markeredgecolor='#000000', 
                 transform=proj)
        
        ax1.legend(handles=[res1, res2, res3], ncol=1,
                            borderaxespad=0, frameon=True, 
                            framealpha=0.9)

        ax1.set_title("Alert levels at POIs - {0}".format(key), fontsize=16)
        # ax1.set_xlabel(r'Longitude ($^\circ$)', fontsize=14)
        # ax1.set_ylabel(r'Latitude ($^\circ$)', fontsize=14)
        ab1 = AnnotationBbox(im, (0, 0), xycoords='axes fraction', box_alignment=(-0.12,-0.3))
        ax1.add_artist(ab1)

        # ax2: plot Al at FCPs  
        # ax2 = plt.axes(projection=cartopy.crs.Mercator())
        ax2.add_feature(coastline, edgecolor='#000000', facecolor='#cccccc', linewidth=1)
        ax2.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
        ax2.add_feature(cartopy.feature.STATES.with_scale('50m'))
        ax2.add_feature(cartopy.feature.OCEAN.with_scale('50m'))

        gl = ax2.gridlines(crs=proj, draw_labels=True, linewidth=1,
                           color="#ffffff", alpha=0.5, linestyle='-')
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

        ax2.scatter(fcp[:,0], fcp[:,1], c=al2, s=40, marker="^", 
                    linewidths=0.75, edgecolors="#000000", label=" ",
                    cmap=cmap, clip_on=True, transform=proj, zorder=10, norm=norm)

        ax2.plot(ev_lon, ev_lat, linewidth=0, marker='*', markersize=14, 
                 markerfacecolor='white', markeredgecolor='#000000', 
                 transform=proj)

        ax2.legend(handles=[res1, res2, res3], ncol=1,
                            borderaxespad=0, frameon=True, 
                            framealpha=0.9)

        ax2.set_title("Alert levels at FCPs - {0}".format(key), fontsize=16)
        # ax2.set_xlabel(r'Longitude ($^\circ$)', fontsize=14)
        # ax2.set_ylabel(r'Latitude ($^\circ$)', fontsize=14)
        ab2 = AnnotationBbox(im, (0, 0), xycoords='axes fraction', box_alignment=(-0.12,-0.3))
        ax2.add_artist(ab2)

        # ax.axis("equal")
        # ax.set_xlim(-5.0, 16.0)
        # ax.set_ylim(32.0, 48.0)
        # ax.set_extent([-8.0, 38.0, 28.0, 48.0], crs=proj)
        # ax.set_extent([lon-5.5, lon+5.5, lat-4.5, lat+4.5], crs=proj)

        outfile_map = os.path.join(fdir, fname + '_' + key + '.png')
        plt.savefig(outfile_map, format='png', dpi=150, bbox_inches='tight')
        plt.close()


def main(**kwargs):

    workflow_dict = kwargs.get('workflow_dict', None)             # workflow dictionary
    event_dict    = kwargs.get('event_dict', None)                # event dictionary
    pois_d = kwargs.get('pois_d', None)
    fcp_d  = kwargs.get('fcp', None)
    hc = kwargs.get('hc_pois', None)
    hc_d = kwargs.get('hc_d', None)
    pois_al = kwargs.get('pois_al', None)
    fcp_al = kwargs.get('fcp_al', None)
    logo = kwargs.get('logo_png', None)             # workflow dictionary
    thresholds = kwargs.get('thresholds', None)
    logger = kwargs.get('logger', None)

    workdir = workflow_dict['workdir']
    percentiles = workflow_dict['percentiles']

    # create plot folder
    figures_dir = os.path.join(workdir, workflow_dict['step5_figures'])
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    # pois_idx = pois_d['pois_index']
    # pois_coords = pois_d['pois_coords'][pois_idx]
    pois_coords = pois_d['pois_coords']

    fcp = np.array([fcp_d[key][1] for key in fcp_d.keys()])

    if hc is None:
        file_hc = os.path.join(workdir, workflow_dict['step3_hc'] + '.npz')
        try:
            hc_sparse = scipy.sparse.load_npz(file_hc)
            hc = hc_sparse.toarray()
        except:
            raise Exception(f"Error reading file: {file_hc}")

    # n_pois, n_thr = hc.shape
    if hc_d is None:
        file_hc_d = os.path.join(workdir, workflow_dict['step3_hc_perc'] + '.npy')
        try:
            hc_d = np.load(file_hc_d, allow_pickle=True).item()
        except:
            raise Exception(f"Error reading file: {file_hc_d}")
    
    if pois_al is None:
        pois_al = np.load(os.path.join(workdir, workflow_dict['step4_alert_levels_POI']),
                        allow_pickle=True).item()

    if fcp_al is None:
        fcp_al = np.load(os.path.join(workdir, workflow_dict['step4_alert_levels_FCP']),
                        allow_pickle=True).item()

    # logger.info('Plotting Alert Levels at POIs')
    # plot_alert_maps(pois_coords, pois_al, event_dict, "POIs", logo, 
    #                 figures_dir, workflow_dict['step5_alert_levels'], logger)

    # logger.info('Plotting Alert Levels at FCPs')
    # plot_alert_maps(fcp, fcp_al, event_dict, "FCPs", logo,
    #                 figures_dir, workflow_dict['step5_alert_levels'], logger)

    logger.info('Plotting Alert Levels at POIs and FCPs')
    plot_alert_maps(pois_coords, pois_al, fcp, fcp_al, event_dict, logo, 
                    figures_dir, workflow_dict['step5_alert_levels'], logger)
    
    logger.info('Plotting hazard maps')
    plot_hazard_maps(pois_coords, hc_d, event_dict, "HazMap", logo,
                     figures_dir, workflow_dict['step5_hazard_maps'], logger)
    

if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]))

# old function
# def plot_alert_maps(points, alert_levels, event_dict, points_label, logo, fdir, fname, logger):
#     """
#     """
#     
#     proj = cartopy.crs.PlateCarree()
#     cmap = matplotlib.colors.ListedColormap(['#00000000', '#39ff00', '#edd400', '#ff0000'])
#     bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
#     norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
#     ev_lon = event_dict['lon']
#     ev_lat = event_dict['lat']
#     ev_depth = event_dict['depth']
#     ev_mag = event_dict['mag']
#     ev_place = event_dict['place']
# 
#     for key, alert_level in alert_levels.items():
# 
#         logger.info("mapping ... {}".format(key))
#         fig = plt.figure(figsize=(16, 8))
#         ax = plt.axes(projection=cartopy.crs.Mercator())
#         coastline = cartopy.feature.GSHHSFeature(scale='low', levels=[1])
#         #coastline = cartopy.feature.GSHHSFeature(scale='high', levels=[1])
#         ax.add_feature(coastline, edgecolor='#000000', facecolor='#cccccc', linewidth=1)
#         ax.add_feature(cartopy.feature.BORDERS.with_scale('50m'))
#         ax.add_feature(cartopy.feature.STATES.with_scale('50m'))
#         ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'))
#         gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1,
#                            color="#ffffff", alpha=0.5, linestyle='-')
#         gl.top_labels = False
#         gl.right_labels = False
#         gl.bottom_labels = True
#         gl.left_labels = True
#         gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
#         gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
#         gl.xlabel_style = {'size': 14}
#         gl.ylabel_style = {'size': 14}
# 
#         if points_label == "POIs":
#             ax.scatter(points[:,0], points[:,1], c=alert_level, s=17, marker="o", 
#                         linewidths=0.75, edgecolors="#000000", label=" ",
#                         cmap=cmap, clip_on=True, transform=proj, zorder=10, norm=norm)
#         elif points_label == "FCPs":
#             ax.scatter(points[:,0], points[:,1], c=alert_level, s=40, marker="^", 
#                         linewidths=0.75, edgecolors="#000000", label=" ",
#                         cmap=cmap, clip_on=True, transform=proj, zorder=10, norm=norm)
#         else:
#             raise Exception('Error in points_label variable:', points_label)
#             #sys.exit('Error in points_label variable:', points_label)
# 
#         ax.plot(ev_lon, ev_lat, linewidth=0, marker='*', markersize=14, 
#                 #markerfacecolor='#c0bfbc', markeredgecolor='#000000', 
#                 markerfacecolor='magenta', markeredgecolor='#000000', 
#                 transform=proj)
# 
#         res1 = matplotlib.patches.Patch(color='#ff0000',
#                                         label=r'Watch',
#                                         alpha=1)
#         res2 = matplotlib.patches.Patch(color='#edd400',
#                                         label=r'Advisory',
#                                         alpha=1)
#         res3 = matplotlib.patches.Patch(color='#39ff00',
#                                         label=r'Information',
#                                         alpha=1)
# 
#         ax.legend(handles=[res1, res2, res3], ncol=1,
#                                borderaxespad=0, frameon=True, 
#                                framealpha=0.9)
# 
#         # ax.set_title("Alert levels at {0} - {1}".format(points_label, key))
#         plt.suptitle("Alert levels at {0} - {1}".format(points_label, key),fontsize=24)
#         plt.title("Epicentral Region: {0} \n Event parameters: Lon={1}, Lat={2}, Depth={3}; Magnitude={4}".format(ev_place, ev_lon, ev_lat, ev_depth, str(ev_mag)[:3] ),fontsize=18)
#         # ax.legend(loc="upper left")
#         ax.set_xlabel(r'Longitude ($^\circ$)', fontsize=14)
#         ax.set_ylabel(r'Latitude ($^\circ$)', fontsize=14)
#         # ax.axis("equal")
#         # ax.set_xlim(-5.0, 16.0)
#         # ax.set_ylim(32.0, 48.0)
#         # ax.set_extent([-8.0, 38.0, 28.0, 48.0], crs=proj)
#         # ax.set_extent([lon-5.5, lon+5.5, lat-4.5, lat+4.5], crs=proj)
# 
#         arr_img = plt.imread(logo)
#         im = OffsetImage(arr_img, zoom=0.1)
#         ab = AnnotationBbox(im, (0, 0), xycoords='axes fraction', box_alignment=(-0.12,-0.3))
#         ax.add_artist(ab)
# 
#         outfile_map = os.path.join(fdir, fname + '_' + key + '_' + points_label + '.png')
#         plt.savefig(outfile_map, format='png', dpi=150, bbox_inches='tight')
