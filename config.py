import torch
import argparse
import sys
import time
import random
import wandb
import numpy as np


def get_run_script():
    run_script = 'python'
    for e in sys.argv:
        run_script += (' ' + e)
    return run_script


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_arg_parser():
    # Arguments
    parser = argparse.ArgumentParser()

    # Add arguments below
    base_args = parser.add_argument_group('Base args')
    base_args.add_argument('--run_script')
    base_args.add_argument('--gpu', type=int, default=0)
    base_args.add_argument('--debug_mode', type=str2bool, default=0)
    base_args.add_argument('--run_mode', type=str, default='train')
    base_args.add_argument('--data_path', type=str, default='/shared/tf_data')
    base_args.add_argument('--dataset', type=str, default='wmt14')
    base_args.add_argument('--machine_name', type=str)

    wandb_args = parser.add_argument_group('wandb args')
    wandb_args.add_argument('--wandb', type=str2bool, default=0)
    wandb_args.add_argument('--project', type=str)
    wandb_args.add_argument('--name', type=str, default='test')
    wandb_args.add_argument('--tags')

    train_args = parser.add_argument_group('Train args')
    train_args.add_argument('--random_seed', type=int, default=1234)
    train_args.add_argument('--n_epochs', type=int, default=1000)
    train_args.add_argument('--batch_size', type=int, default=32)
    train_args.add_argument('--eval_steps', type=int, default=1000)
    train_args.add_argument('--max_len', type=int, default=100)
    train_args.add_argument('--min_freq', type=int, default=2)
    train_args.add_argument('--beam_size', type=int, default=3)
    train_args.add_argument('--length_penalty', type=float, default=0.6)
    train_args.add_argument('--decode_max_len_add', type=int, default=50)

    network_args = parser.add_argument_group('Network args')
    network_args.add_argument('--n_layers', type=int, default=6)
    network_args.add_argument('--n_heads', type=int, default=8)
    network_args.add_argument('--d_model', type=int, default=512)
    network_args.add_argument('--p_dropout', type=float, default=0.1)
    network_args.add_argument('--enc_self_attn', type=str, default='vanilla')
    network_args.add_argument('--dec_self_attn', type=str, default='vanilla')
    network_args.add_argument('--cross_attn', type=str, default='vanilla')
    network_args.add_argument('--attn_norm', type=str, default='softmax')
    network_args.add_argument('--sparsity_mode', type=str, default='none')
    network_args.add_argument('--sparsity_top_k', type=int, default=8)

    return parser


def get_args():
    parser = make_arg_parser()
    args = parser.parse_args()
    args.run_script = get_run_script()

    # set pytorch print option
    torch.set_printoptions(threshold=10000)

    # random_seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    args.tags = [e for e in args.tags.split(',')] if args.tags is not None else ['test']
    args.tags.append(args.name)

    # parse gpus
    if args.gpu < 0:
        args.device = 'cpu'
    else:
        args.device = f'cuda:{args.gpu}'

    if args.dataset not in ['multi30k', 'iwslt', 'wmt14']:
        assert False

    if args.debug_mode:
        args.data_path = '.data_debug'

    if args.wandb:
        args.project = f'efficient_transformer_{args.dataset}'
        wandb.init(project=args.project, name=args.name, tags=args.tags, config=args)

    if args.enc_self_attn not in ['vanilla', 'dense', 'random', 'dense_random']:
        assert False

    if args.dec_self_attn not in ['vanilla', 'dense', 'random', 'dense_random']:
        assert False

    if args.cross_attn not in ['vanilla', 'dense', 'random', 'dense_random']:
        assert False

    if args.attn_norm not in ['softmax', 'tanh', 'sigmoid']:
        assert False

    if args.sparsity_mode not in ['topk', 'none']:
        assert False

    return args, parser


def print_args(args):
    info = '\n[args]\n'
    for sub_args in parser._action_groups:
        if sub_args.title in ['positional arguments', 'optional arguments']:
            continue
        size_sub = len(sub_args._group_actions)
        info += f'  {sub_args.title} ({size_sub})\n'
        for i, arg in enumerate(sub_args._group_actions):
            prefix = '-'
            info += f'      {prefix} {arg.dest:20s}: {getattr(args, arg.dest)}\n'
    info += '\n'
    print(info)


ARGS, parser = get_args()
if ARGS.debug_mode:
    print_args(ARGS)


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
