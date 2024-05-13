class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/whuai/tracker_code/DeT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = ''
        self.got10k_dir = '/home/whuai/dataset_track/GOT-10K/train/color'
        self.trackingnet_dir = ''
        self.coco_dir = '/home/whuai/dataset_track/COCO/images/train2017'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.cdtb_dir = ''
        self.depthtrack_dir = '/home/whuai/dataset_track/depthtrack/train/'
        self.lasotdepth_dir = ''
        self.cocodepth_dir = '/home/whuai/dataset_track/COCO/'
