import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import os

from isegm.inference import utils
from isegm.utils.exp import load_config_file
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import yaml
from isegm.inference.evaluation import evaluate_dataset

# Lazy imports for modules only needed in specific modes
def _get_sam_predictor():
    from isegm.inference.predictors import get_predictor, SamPredictor
    return get_predictor, SamPredictor

def _get_sam_model_registry():
    from isegm.model.build_sam import sam_model_registry
    return sam_model_registry

def _get_pos_embed():
    from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference
    return interpolate_pos_embed_inference

def _get_vis_utils():
    from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks
    return draw_probmap, draw_with_blend_and_clicks

os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'

os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '0'

os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_DESYNC_DEBUG'] = '1'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'mode',
        choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
        help='')

    group_checkpoints = parser.add_mutually_exclusive_group(required=True)
    group_checkpoints.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='The path to the checkpoint. '
        'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
        'or an absolute path. The file extension can be omitted.')
    group_checkpoints.add_argument(
        '--exp-path',
        type=str,
        default='',
        help='The relative path to the experiment with checkpoints.'
        '(relative to cfg.EXPS_PATH)')

    parser.add_argument(
        '--datasets',
        type=str,
        default='GrabCut,Berkeley,SBD,DAVIS',
        help='List of datasets on which the model should be tested. '
        'Datasets are separated by a comma. Possible choices: '
        'GrabCut, Berkeley, DAVIS, SBD, PascalPart, PartImageNet, SA1B, BraTS, ssTEM, OAIZIB')

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0', help='ID of used GPU.')
    group_device.add_argument(
        '--cpu', action='store_true', default=False, help='Use only CPU for inference.')

    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument(
        '--target-iou',
        type=float,
        default=0.90,
        help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    group_iou_thresh.add_argument(
        '--iou-analysis',
        action='store_true',
        default=False,
        help='Plot mIoU(number of clicks) with target_iou=1.0.')

    parser.add_argument(
        '--n-clicks', type=int, default=20, help='Maximum number of clicks for the NoC metric.')
    parser.add_argument(
        '--min-n-clicks', type=int, default=1, help='Minimum number of clicks for the evaluation.')
    parser.add_argument(
        '--thresh',
        type=float,
        required=False,
        default=0.5,
        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--clicks-limit', type=int, default=None)
    parser.add_argument(
        '--eval-mode',
        type=str,
        default='cvpr',
        help='Choices: cvpr, fixed<number>, or fixed<number>,<number>,(e.g. fixed400, fixed400,600).'
    )

    parser.add_argument('--eval-ritm', action='store_true', default=False)
    parser.add_argument('--save-ious', action='store_true', default=False)
    parser.add_argument('--print-ious', action='store_true', default=False)
    parser.add_argument('--vis-preds', action='store_true', default=False)
    parser.add_argument('--class-name', type=str, default=None)
    parser.add_argument('--part-name', type=str, default=None)
    parser.add_argument('--graco', action='store_true', default=False)
    parser.add_argument('--gra', type=float, default=None, help='Granularity slider')
    parser.add_argument('--phrase', type=str, default=None, help='Semantic phrase')
    parser.add_argument(
        '--granularities',
        type=str,
        default=None,
        help='Comma-separated list of granularities to test, e.g., "0.1,0.5,1.0". Default: 0.1,0.2,...,1.0')
    parser.add_argument('--lora_checkpoint', type=str, default=None)
    
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='The model name that is used for making plots.')
    parser.add_argument(
        '--sam-model',
        type=str,
        default=None,
        help='The Segment Anything model name that is used for inference.')
    parser.add_argument(
        '--sam-type',
        type=str,
        default=None,
        help='The Segment Anything model type that is used for inference.')
    parser.add_argument(
        '--sam-cfg-path',
        type=str,
        default=None,
        help='config file for SAM that is used for inference.')
    # parser.add_argument('--sam2-cfg', action='store_true', default=False)
    parser.add_argument('--oracle', action='store_true', default=False)
    parser.add_argument(
        '--config-path', type=str, default='./config.yml', help='The path to the config file.')
    parser.add_argument(
        '--logs-path',
        type=str,
        default='',
        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')

    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--distributed', action='store_true', default=False, help='Use distributed evaluation')

    args = parser.parse_args()
    
    if args.granularities:
        args.granularities_list = [float(g.strip()) for g in args.granularities.split(',')]
    else:
        args.granularities_list = None
    
    if args.distributed and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        print(f"Setting local_rank from environment: {args.local_rank}")
    
    if args.cpu:
        args.device = torch.device('cpu')
    elif args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device(f'cuda:{args.local_rank}')
        print(f"Rank {args.local_rank}: Using device {args.device}")
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)

    return args, cfg


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        print(f"Distributed training: rank={args.rank}, world_size={args.world_size}, local_rank={args.local_rank}")
    else:
        print("Not using distributed training - missing environment variables")
        args.distributed = False
        return

    args.distributed = True
    
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.distributed = False
        return
        
    if args.local_rank >= torch.cuda.device_count():
        print(f"Error: local_rank {args.local_rank} >= available GPUs {torch.cuda.device_count()}")
        args.distributed = False
        return
    
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'
    
    print(f'Rank {args.rank}: Initializing distributed training on device cuda:{args.local_rank}')
    
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    
    current_device = torch.cuda.current_device()
    print(f"Rank {args.rank}: Current CUDA device after init: {current_device}")
    
    torch.distributed.barrier(device_ids=[args.local_rank])
    
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def main():
    args, cfg = parse_args()

    if args.distributed:
        init_distributed_mode(args)
        
        if args.distributed:
            print(f"Rank {args.rank}: Final device check - using {args.device}")
            assert torch.cuda.current_device() == args.local_rank, f"Device mismatch: current={torch.cuda.current_device()}, expected={args.local_rank}"
    
    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    if args.distributed and args.rank == 0 or not args.distributed:
        logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = len(checkpoints_list) == 1
    assert not args.iou_analysis if not single_model_eval else True, \
        "Can't perform IoU analysis for multiple checkpoints"
    print_header = single_model_eval
    target_iou = args.target_iou
    
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg, args)

        for checkpoint_path in checkpoints_list:
            print(f"Rank {getattr(args, 'rank', 0)}: Loading model on device {args.device}")
            
            # for sam
            if args.sam_type == 'SAM':
                assert args.sam_model in ['vit_b', 'vit_l', 'vit_h']
                sam_model_registry = _get_sam_model_registry()
                get_predictor, SamPredictor = _get_sam_predictor()
                sam = sam_model_registry[args.sam_model](checkpoint=checkpoint_path)
                sam.to(device=args.device)
                predictor = SamPredictor(sam)

            elif args.sam_type == 'SAM2':
                model_config = args.sam_cfg_path
                sam2_model = build_sam2(model_config, args.checkpoint, device=args.device, strict=False)
                predictor = SAM2ImagePredictor(sam2_model, mask_threshold=-1, max_hole_area=0, max_sprinkle_area=0)
                print(f"Rank {getattr(args, 'rank', 0)}: SAM2 model loaded on {args.device}")
                
            else:
                get_predictor, SamPredictor = _get_sam_predictor()
                interpolate_pos_embed_inference = _get_pos_embed()
                model = utils.load_is_model(checkpoint_path, args.device, args.eval_ritm, args.lora_checkpoint)
                predictor_params, zoomin_params = get_predictor_and_zoomin_params(
                    args, dataset_name, eval_ritm=args.eval_ritm)
                if not args.eval_ritm:
                    interpolate_pos_embed_inference(model.backbone, zoomin_params['target_size'],
                                                    args.device)
                predictor = get_predictor(
                    model,
                    args.mode,
                    args.device,
                    prob_thresh=args.thresh,
                    predictor_params=predictor_params,
                    zoom_in_params=zoomin_params)

            vis_callback = get_prediction_vis_callback(logs_path, dataset_name,
                                                       args.thresh) if args.vis_preds else None
            if dataset_name == 'PascalPart' or dataset_name == 'PartImageNet':
                args.target_iou = 0.85 # only eval NoC@85 for Part-level Dataset
            else:
                args.target_iou = target_iou
                
            print(dataset_name)
            use_distributed = args.distributed and dataset_name != 'SA1B'
            if use_distributed:
                print("Using distributed evaluation")
                dataset_results = evaluate_dataset(
                    dataset,
                    predictor,
                    graco=args.graco,
                    sam_type=args.sam_type,
                    oracle=args.oracle,
                    gra=args.gra,
                    phrase=args.phrase,
                    granularities=args.granularities_list,
                    pred_thr=args.thresh,
                    max_iou_thr=args.target_iou,
                    min_clicks=args.min_n_clicks,
                    max_clicks=args.n_clicks,
                    callback=vis_callback,
                    distributed=True)
            else:
                # SA1B or non-distributed configuration use single-process evaluation
                dataset_results = evaluate_dataset(
                    dataset,
                    predictor,
                    graco=args.graco,
                    sam_type=args.sam_type,
                    oracle=args.oracle,
                    gra=args.gra,
                    phrase=args.phrase,
                    granularities=args.granularities_list,
                    pred_thr=args.thresh,
                    max_iou_thr=args.target_iou,
                    min_clicks=args.min_n_clicks,
                    max_clicks=args.n_clicks,
                    callback=vis_callback,
                    distributed=False)

            if not args.distributed or args.rank == 0:
                row_name = args.mode if single_model_eval else checkpoint_path.stem
                if args.iou_analysis:
                    save_iou_analysis_data(
                        args,
                        dataset_name,
                        logs_path,
                        logs_prefix,
                        dataset_results,
                        model_name=args.model_name)

                save_results(
                    args,
                    row_name,
                    dataset_name,
                    logs_path,
                    logs_prefix,
                    dataset_results,
                    save_ious=single_model_eval and args.save_ious,
                    single_model_eval=single_model_eval,
                    print_header=print_header)
                print_header = False


def get_predictor_and_zoomin_params(args, dataset_name, apply_zoom_in=True, eval_ritm=False):
    predictor_params = {}

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    zoom_in_params = None
    if apply_zoom_in and eval_ritm:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {'target_size': 600 if dataset_name == 'DAVIS' else 400}
        elif args.eval_mode.startswith('fixed'):
            crop_size = int(args.eval_mode[5:])
            zoom_in_params = {'skip_clicks': -1, 'target_size': (crop_size, crop_size)}
        else:
            raise NotImplementedError

    if apply_zoom_in and not eval_ritm:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (672, 672) if dataset_name == 'DAVIS' else (448, 448)
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = args.eval_mode.split(',')
            crop_size_h = int(crop_size[0][5:])
            crop_size_w = crop_size_h
            if len(crop_size) == 2:
                crop_size_w = int(crop_size[1])
            zoom_in_params = {'skip_clicks': -1, 'target_size': (crop_size_h, crop_size_w)}
        else:
            raise NotImplementedError

    return predictor_params, zoom_in_params


def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''
    if args.exp_path:
        rel_exp_path = args.exp_path
        checkpoint_prefix = ''
        if ':' in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')

        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
        assert len(candidates) == 1, 'Invalid experiment path.'
        exp_path = candidates[0]
        checkpoints_list = sorted(
            (exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pth'), reverse=True)
        assert len(checkpoints_list) > 0, "Couldn't find any checkpoints."

        if checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f'all_{checkpoint_prefix}'
        else:
            logs_prefix = 'all_checkpoints'

        logs_path = args.logs_path / exp_path.relative_to(cfg.EXPS_PATH)
    else:
        # If checkpoint is an absolute path that exists, use it directly
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.is_absolute() and checkpoint_path.exists():
            checkpoints_list = [checkpoint_path]
        else:
            # Fall back to find_checkpoint with INTERACTIVE_MODELS_PATH
            weights_folder = getattr(cfg, 'INTERACTIVE_MODELS_PATH', './weights')
            checkpoints_list = [
                Path(utils.find_checkpoint(weights_folder, args.checkpoint))
            ]
        logs_path = args.logs_path / 'others' / checkpoints_list[0].stem

    return checkpoints_list, logs_path, logs_prefix


def save_results(args,
                 row_name,
                 dataset_name,
                 logs_path,
                 logs_prefix,
                 dataset_results,
                 save_ious=False,
                 print_header=True,
                 single_model_eval=False):
    all_ious, elapsed_time = dataset_results
    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, noc_list_std, over_max_list = utils.compute_noc_metric(
        all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)
    iou_first = np.array([ious[0] for ious in all_ious]).mean(0)
    row_name = 'last' if row_name == 'last_checkpoint' else row_name
    model_name = str(logs_path.relative_to(
        args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    header, table_row = utils.get_results_table(
        noc_list,
        over_max_list,
        row_name,
        dataset_name,
        mean_spc,
        elapsed_time,
        iou_first,
        args.n_clicks,
        model_name=model_name)

    if args.print_ious:
        min_num_clicks = min(len(x) for x in all_ious)
        mean_ious = np.array([x[:min_num_clicks] for x in all_ious]).mean(axis=0)
        miou_str = ' '.join([
            f'mIoU@{click_id}={mean_ious[click_id - 1]:.2%};'
            for click_id in [_ for _ in range(1, 21)] if click_id <= min_num_clicks
        ])
        table_row += '; ' + miou_str
    else:
        target_iou_int = int(args.target_iou * 100)
        if target_iou_int not in [80, 85, 90]:
            noc_list, _, over_max_list = utils.compute_noc_metric(
                all_ious, iou_thrs=[args.target_iou], max_clicks=args.n_clicks)
            table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
            table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header:
        print(header)
    print(table_row)

    if save_ious:
        ious_path = logs_path / 'ious' / (logs_prefix if logs_prefix else '')
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(ious_path / f'{dataset_name}_{args.eval_mode}_{args.mode}_{args.n_clicks}.pkl',
                  'wb') as fp:
            pickle.dump(all_ious, fp)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
        if not single_model_eval:
            name_prefix += f'{dataset_name}_'

    log_path = logs_path / f'{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.txt'
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            if print_header:
                f.write(header + '\n')
            f.write(table_row + '\n')


def save_iou_analysis_data(args,
                           dataset_name,
                           logs_path,
                           logs_prefix,
                           dataset_results,
                           model_name=None):
    all_ious, _ = dataset_results

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
    name_prefix += dataset_name + '_'
    if model_name is None:
        model_name = str(logs_path.relative_to(
            args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem

    pkl_path = logs_path / f'plots/{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.pickle'
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump(
            {
                'dataset_name': dataset_name,
                'model_name': f'{model_name}_{args.mode}',
                'all_ious': all_ious
            }, f)


def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    draw_probmap, draw_with_blend_and_clicks = _get_vis_utils()

    def callback(image, gt_mask, pred_probs, sample_id, click_indx, clicks_list, gra):
        sample_path = save_path / f'{gra}'
        sample_path.mkdir(parents=True, exist_ok=True)
        sample_path = save_path / f'{gra}' / f'{sample_id}_{click_indx}.jpg'
        prob_map = draw_probmap(pred_probs)
        image_with_mask = draw_with_blend_and_clicks(
            image, pred_probs > 0, clicks_list=clicks_list)
        cv2.imwrite(
            str(sample_path),
            np.concatenate((image_with_mask, prob_map), axis=1)[:, :, ::-1])

    return callback


if __name__ == '__main__':
    main()
