from datetime import timedelta
from pathlib import Path
import torch
import numpy as np

from isegm.data.datasets import *

# Lazy import for load_model (only needed for non-SAM/SAM2 models)
def _get_load_model():
    from isegm.utils.serialization import load_model
    return load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    if n_clicks == 0:
        mean_spc = 0
    else: 
        mean_spc = elapsed_time / n_clicks
    if n_images == 0:
        mean_spi = 0
    else:
        mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, eval_ritm, lora_checkpoint=None, config=None, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
        # print("Load pre-trained checkpoint from: %s" % checkpoint)
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, eval_ritm, **kwargs)
        models = [load_single_is_model(x, device, eval_ritm, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, eval_ritm, lora_checkpoint=lora_checkpoint, config=config, **kwargs)


def load_single_is_model(state_dict, device, eval_ritm, lora_checkpoint=None, config=None, **kwargs):
    if config:
        _config = config
    else:
        _config = state_dict['config']
    if lora_checkpoint is not None:
        lora_state_dict = torch.load(lora_checkpoint, map_location='cpu')
        _config = lora_state_dict['config']

    load_model = _get_load_model()
    model = load_model(_config, eval_ritm, **kwargs)
    msg = model.load_state_dict(state_dict['state_dict'], strict=False)

    if lora_checkpoint is not None:
        msg = model.load_state_dict(lora_state_dict['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_dataset(dataset_name, cfg, args):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train')
    elif dataset_name == 'PascalPart':
        dataset = PascalPartEvaluationDataset(cfg.PASCALPART_PATH, split='val', class_name=args.class_name,
                                              part_name=args.part_name)
    elif dataset_name == 'PartImageNet':
        dataset = PartINEvaluationDataset(cfg.PARTIMAGENET_PATH, split='val', class_name=args.class_name,
                                          part_name=args.part_name)
    elif dataset_name == 'SA1B':
        dataset = SA1BDataset(cfg.SA1B_PATH)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    noc_list_std = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int_)

        score = scores_arr.mean()
        score_std = scores_arr.std()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        noc_list_std.append(score_std)
        over_max_list.append(over_max)

    return noc_list, noc_list_std, over_max_list


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pt'): # Junwei: before was .pth
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pt'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time, iou_first,
                      n_clicks=20, model_name=None):
    table_header = (f'|{"BRS Type":^13}|{"Dataset":^11}|'
                    f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|'
                    f'{"IoU@1":^9}|'
                    f'{">="+str(n_clicks)+"@85%":^9}|{">="+str(n_clicks)+"@90%":^9}|'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|'
    table_row += f'{noc_list[0]:^9.4f}|'
    table_row += f'{noc_list[1]:^9.4f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.4f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{iou_first:^9.4f}|'
    table_row += f'{over_max_list[1]:^9}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.4f}|{eval_time:^9}|'

    return header, table_row
