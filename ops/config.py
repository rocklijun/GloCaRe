import argparse

import json
def parse_list(arg):
    return json.loads(arg)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str,
                    # required=True
                    )
parser.add_argument('--unlabeled-id-path', type=str)
parser.add_argument('--save-path', type=str,
                    # required=True
                    )
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

# test
parser.add_argument('--checkpoint', type=str, help="Path to checkpoint (pth file)")
parser.add_argument('--save_vis_dir', type=str)
parser.add_argument('--mode', type=str)

parser.add_argument('--use_rff', default=True, type=str2bool)
parser.add_argument('--ratio_rff_dim', default=4, type=int)



# for number of fourier spaces
parser.add_argument ('--num_f', type=int, default=1, help = 'number of fourier spaces')

parser.add_argument ('--sample_rate', type=float, default=1.0, help = 'sample ratio of the features involved in balancing')
parser.add_argument ('--lrbl', type = float, default = 1.0, help = 'learning rate of balance')

# parser.add_argument ('--cfs', type = int, default = 512, help = 'the dim of each feature')
parser.add_argument ('--lambdap', type = float, default = 70.0, help = 'weight decay for weight1 ')
parser.add_argument ('--lambdapre', type = float, default = 1, help = 'weight for pre_weight1 ')

parser.add_argument ('--epochb', type = int, default = 20, help = 'number of epochs to balance')
parser.add_argument ('--epochp', type = int, default = 0, help = 'number of epochs to pretrain')

parser.add_argument ('--n_feature', type=int, default=12, help = 'number of pre-saved features')
parser.add_argument ('--feature_dim', type=int, default=512, help = 'the dim of each feature')

parser.add_argument ('--lrwarmup_epo', type=int, default=0, help = 'the dim of each feature')
parser.add_argument ('--lrwarmup_decay', type=int, default=0.1, help = 'the dim of each feature')

parser.add_argument ('--n_levels', type=int, default=1, help = 'number of global table levels')

# for expectation
parser.add_argument ('--lambda_decay_rate', type=float, default=1, help = 'ratio of epoch for lambda to decay')
parser.add_argument ('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
parser.add_argument ('--min_lambda_times', type=float, default=0.01, help = 'number of global table levels')

# for jointly train
parser.add_argument ('--train_cnn_with_lossb', type=bool, default=False, help = 'whether train cnn with lossb')
parser.add_argument ('--cnn_lossb_lambda', type=float, default=0, help = 'lambda for lossb')

# for more moments
parser.add_argument ('--moments_lossb', type=float, default=1, help = 'number of moments')

# for first step
parser.add_argument ('--first_step_cons', type=float, default=1, help = 'constrain the weight at the first step')

# for pow
parser.add_argument ('--decay_pow', type=float, default=2, help = 'value of pow for weight decay')

# for second order moment weight
parser.add_argument ('--second_lambda', type=float, default=0.2, help = 'weight lambda for second order moment loss')
parser.add_argument ('--third_lambda', type=float, default=0.05, help = 'weight lambda for second order moment loss')

# for dat/.a aug
parser.add_argument ('--lower_scale', type=float, default=0.8, help = 'weight lambda for second order moment loss')

# for lr decay epochs
parser.add_argument ('--epochs_decay', type=parse_list, default=[24, 30], help = 'weight lambda for second order moment loss')


parser.add_argument ('--gray_scale', type=float, default=0.1, help = 'weight lambda for second order moment loss')

parser.add_argument('--sum', type=bool, default=True, help='sum or concat')
parser.add_argument('--concat', type=int, default=1, help='sum or concat')
parser.add_argument('--min_scale', type=float, default=0.8, help='')
parser.add_argument('--presave_ratio', type=float, default=0.9, help='the ratio for presaving features')


# for schedule.py
parser.add_argument('--lr', '--learning-rate', default=0.000005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
