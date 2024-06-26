import os
import sys
import argparse
envpath = '/home/whuai/mambaforge/envs/Det-1/lib/python3.7/site-packages/cv2/qt/plugins'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker


def run_tracker(tracker_name, tracker_dir, tracker_param, run_id=None, dataset_name='cdtb', dtype='rgb', sequence=None, debug=0, threads=0,
                visdom_info=None, use_vot=False):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info
    dataset = get_dataset(dtype, dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(name=tracker_name,parameter_dir= tracker_dir,parameter_name=tracker_param,run_id=run_id, use_vot=use_vot)]

    run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info, run_id=run_id) # Song add run_id for box_file saving


'''  One more parameter,
        --input_dtype : including rgb, rgbcolormap, colormap, raw_depth, normalized_depth, rgb_rawdepth, rgb_normdepth,
                        here, we use the rgbcolormap, which includes the rgb + colormap of depth images,


    CUDA_VISIBLE_DEVICES=0 python run_tracker.py dimp DeT_DiMP50_Max --dataset_name depthtrack --input_dtype rgbcolormap --sequence adapter01_indoor --debug 1

    CUDA_VISIBLE_DEVICES=0 python run_tracker.py dimp DeT_DiMP50_Max --dataset_name depthtrack --input_dtype rgbcolormap

python run_tracker.py dimp cdtb_cocograd1_Mean_test mycocoDeT_DiMP50_Mean_51 cdtb --usevot True
    
'''
def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, default='dimp',help='Name of tracking method.')
    parser.add_argument('tracker_dir', type=str, default='DeT_DiMP50_Mean', help='Name of parameter file.')
    parser.add_argument('tracker_param', type=str, default='DeT_DiMP50_Mean', help='Name of parameter file.')
    parser.add_argument('dataset_name', type=str, default='cdtb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--use_vot',type=bool,default=True,help='choose how to save result depend on vot')
    parser.add_argument('--input_dtype', type=str, default='rgbcolormap', help='[colormap, raw depth, normalized_depth, ....]')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=False, help='Flag to enable visdom.')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name,args.tracker_dir, args.tracker_param, args.runid, args.dataset_name, args.input_dtype, seq_name, args.debug,
                args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port},args.use_vot)


if __name__ == '__main__':
    main()
