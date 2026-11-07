
import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='MUSE time series forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--is_transfering', type=int, required=False, default=0, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='MUSE', choices=['MUSE'], help='model name')
parser.add_argument('--dlf_enabled', type=int, default=1,
                    help='enable DLF (dual-domain latent); 0 uses the base forecast and bypasses CPRC/AMDF')
parser.add_argument('--cprc_enabled', type=int, default=0,
                    help='enable CPRC (low-rank periodic residual correction)')
parser.add_argument('--amdf_enabled', type=int, default=0,
                    help='enable AMDF (adaptive multi-period deviation field and phase alignment)')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# MUSE
parser.add_argument('--alpha', type=float, default=0.5, help='time-domain mixing weight')
parser.add_argument('--lpf', type=int, default=15, help='number of retained Fourier coefficients')
parser.add_argument('--cprc_cap', type=float, default=0.05,
                    help='shared normalized correction bound for CPRC and AMDF')
parser.add_argument('--phase_max_shift', type=int, default=3,
                    help='maximum integer phase shift searched in each historical period')

# Periodic layout and dataset time encoding
parser.add_argument('--period_len', type=int, default=24, help='base period length')
parser.add_argument('--enc_in', type=int, default=7, help='number of input channels')
parser.add_argument('--embed', type=str, default='learned', help='timeF or learned time encoding')
parser.add_argument('--do_predict', action='store_true', help='predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3
                    , help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.02, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=int, default=0, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, help='use multiple gpus', default=0)
parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed_list = range(2023, 2033)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        random.seed(fix_seed_list[ii])
        torch.manual_seed(fix_seed_list[ii])
        np.random.seed(fix_seed_list[ii])
        # setting record of experiments
        # Fixed CPRC/AMDF design values are encoded once below, not read from the
        # CLI.  This keeps experiment names reproducible after argument cleanup.
        setting = '{}_{}_{}_sl{}_per{}_pl{}_a{}_lpf{}_dlf{}_cprc{}_amdf{}_r8_cap{}_k4_shift{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.period_len,
            args.pred_len,
            args.alpha,
            args.lpf,
            args.dlf_enabled,
            args.cprc_enabled,
            args.amdf_enabled,
            args.cprc_cap,
            args.phase_max_shift,
            fix_seed_list[ii])

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:


       ii = 0
       setting = '{}_{}_{}_sl{}_per{}_pl{}_a{}_lpf{}_dlf{}_cprc{}_amdf{}_r8_cap{}_k4_shift{}_seed{}'.format(
           args.model_id,
           args.model,
           args.data,
           args.seq_len,
           args.period_len,
           args.pred_len,
           args.alpha,
           args.lpf,
           args.dlf_enabled,
           args.cprc_enabled,
           args.amdf_enabled,
           args.cprc_cap,
           args.phase_max_shift,
           fix_seed_list[ii])

       exp = Exp(args)  # set experiments
       print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
       exp.test(setting, test=1)
       torch.cuda.empty_cache()
