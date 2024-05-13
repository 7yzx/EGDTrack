from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.cdtb_path = '/media/whuai/Windows-SSD/Users/admin/Desktop/CDTB/sequences'
    settings.davis_dir = ''
    settings.depthtrack_path = '/media/whuai/Windows-SSD/Users/admin/Desktop/dataset/sequences'
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/home/whuai/Tracker_code/DeT/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/home/whuai/dataset_track/OTB100/'
    settings.result_plot_path = '/home/whuai/Tracker_code/DeT/pytracking/result_plots/'
    settings.results_path = '/home/whuai/Tracker_code/DeT/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/whuai/Tracker_code/DeT/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

