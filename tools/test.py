# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--stagea-query-stats',
        action='store_true',
        help='collect the number of valid StageA adaptive queries per sample')
    parser.add_argument(
        '--stagea-query-stats-out',
        help='optional JSON output path for StageA adaptive query statistics')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if args.stagea_query_stats_out:
        args.stagea_query_stats = True
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def register_stagea_query_stats_hook(model, query_counts):
    """Collect valid adaptive-query counts without changing model outputs."""
    head = getattr(model, 'pts_bbox_head', None)
    if head is None:
        raise RuntimeError('The model has no pts_bbox_head for StageA stats.')

    def _collect_query_counts(module, inputs, output):
        if not isinstance(output, dict):
            return
        padding_mask = output.get('far3d_stagea_query_padding_mask')
        if padding_mask is None:
            return
        if padding_mask.ndim != 2:
            raise RuntimeError(
                'Expected a 2D StageA query padding mask, got '
                f'{tuple(padding_mask.shape)}.')
        num_global_queries = int(getattr(module, 'num_query', 0))
        if padding_mask.size(1) < num_global_queries:
            raise RuntimeError(
                'StageA query padding mask is shorter than the global query '
                f'count: {padding_mask.size(1)} < {num_global_queries}.')
        adaptive_padding_mask = padding_mask[:, num_global_queries:]
        counts = (~adaptive_padding_mask.bool()).sum(dim=1)
        query_counts.extend(int(value) for value in counts.detach().cpu())

    return head.register_forward_hook(_collect_query_counts)


def gather_stagea_query_counts(local_counts, dataset_size, distributed):
    """Gather counts in dataset order and remove distributed padding."""
    if not distributed:
        return local_counts[:dataset_size]

    rank, world_size = get_dist_info()
    device = torch.device('cuda', torch.cuda.current_device())
    local_tensor = torch.tensor(local_counts, dtype=torch.long, device=device)
    local_size = torch.tensor(
        [local_tensor.numel()], dtype=torch.long, device=device)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_sizes, local_size)
    sizes = [int(value.item()) for value in gathered_sizes]
    max_size = max(max(sizes), 1)

    padded = torch.full((max_size,), -1, dtype=torch.long, device=device)
    if local_tensor.numel() > 0:
        padded[:local_tensor.numel()] = local_tensor
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, padded)

    if rank != 0:
        return None

    rank_parts = [
        values[:size].cpu().tolist()
        for values, size in zip(gathered, sizes)
    ]
    ordered_counts = []
    for sample_index in range(max_size):
        for part in rank_parts:
            if sample_index < len(part):
                ordered_counts.append(part[sample_index])
    return ordered_counts[:dataset_size]


def summarize_stagea_query_counts(query_counts, query_cap):
    """Build and print StageA adaptive-query statistics."""
    if not query_counts:
        raise RuntimeError(
            'No StageA query counts were collected. Check that StageA is '
            'enabled and the head returns its query padding mask.')

    values = np.asarray(query_counts, dtype=np.int64)
    unique_counts, frequencies = np.unique(values, return_counts=True)
    summary = dict(
        samples=int(values.size),
        mean=float(values.mean()),
        p95=float(np.percentile(values, 95)),
        max=int(values.max()),
        cap=int(query_cap) if query_cap is not None else None,
        cap_hits=None,
        cap_hit_rate=None)
    if query_cap is not None:
        cap_hits = int(np.count_nonzero(values >= query_cap))
        summary['cap_hits'] = cap_hits
        summary['cap_hit_rate'] = float(cap_hits / values.size)

    print('\n[StageA Adaptive Query statistics]')
    print(f"  samples:      {summary['samples']}")
    print(f"  mean:         {summary['mean']:.2f}")
    print(f"  p95:          {summary['p95']:.2f}")
    print(f"  max:          {summary['max']}")
    if query_cap is not None:
        print(f"  cap:          {summary['cap']}")
        print(f"  cap_hits:     {summary['cap_hits']}")
        print(f"  cap_hit_rate: {summary['cap_hit_rate']:.4%}")

    return dict(
        summary=summary,
        histogram={
            str(int(count)): int(frequency)
            for count, frequency in zip(unique_counts, frequencies)
        })


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
       if cfg.plugin:
           import importlib
           if hasattr(cfg, 'plugin_dir'):
               plugin_dir = cfg.plugin_dir
               _module_dir = os.path.dirname(plugin_dir)
               _module_dir = _module_dir.split('/')
               _module_path = _module_dir[0]

               for m in _module_dir[1:]:
                   _module_path = _module_path + '.' + m
               print(_module_path)
               plg_lib = importlib.import_module(_module_path)
           else:
               # import dir is the dirpath for the config file
               _module_dir = os.path.dirname(args.config)
               _module_dir = _module_dir.split('/')
               _module_path = _module_dir[0]
               for m in _module_dir[1:]:
                   _module_path = _module_path + '.' + m
               plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    stagea_query_counts = []
    stagea_stats_hook = None
    if args.stagea_query_stats:
        stagea_stats_hook = register_stagea_query_stats_hook(model, stagea_query_counts)

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    if stagea_stats_hook is not None:
        stagea_stats_hook.remove()
    gathered_query_counts = None
    if args.stagea_query_stats:
        gathered_query_counts = gather_stagea_query_counts(
            stagea_query_counts, len(dataset), distributed)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.stagea_query_stats:
            stagea_cfg = cfg.model.pts_bbox_head.get(
                'far3d_stagea_cfg', {})
            query_cap = stagea_cfg.get('max_adaptive_queries', None)
            query_stats = summarize_stagea_query_counts(
                gathered_query_counts, query_cap)
            if args.stagea_query_stats_out:
                stats_dir = os.path.dirname(args.stagea_query_stats_out)
                if stats_dir:
                    mmcv.mkdir_or_exist(stats_dir)
                mmcv.dump(query_stats, args.stagea_query_stats_out)
                print(
                    '  json:         '
                    f'{args.stagea_query_stats_out}')
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
