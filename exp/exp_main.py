from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import MUSE
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

DLF_FREQ_LOSS_WEIGHT = 1e-4
DLF_FREQ_MASK_LOSS_WEIGHT = 1e-4
DLF_FUTURE_LOSS_WEIGHT = 1e-3
DLF_FUTURE_LPF = 8


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        if self.args.model != 'MUSE':
            raise ValueError('This repository only includes MUSE; use --model MUSE.')

        if self.args.is_transfering and "Electricity" in self.args.model_id:
            self.args.enc_in = 7

        model = MUSE.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):

        if self.args.is_transfering:
            print("Before Transfer, dataset is ", self.args.data)

            if "ETTh1" in self.args.data  :
                self.args.data = "ETTh2"
                self.args.data_path = "ETTh2.csv"


            elif  "Electricity" in self.args.model_id:
                self.args.data = "ETTh2"
                self.args.data_path = "ETTh2.csv"

            print("Transfer to dataset ", self.args.data)

        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _latent_auxiliary_loss(self, aux, future_target=None):
        """Dual-domain reconstruction, consistency, and future-aware supervision."""
        if not getattr(self.args, 'dlf_enabled', 1) or aux is None:
            return 0.0
        loss = 0.0
        if aux['freq_mask'].any():
            loss = loss + DLF_FREQ_MASK_LOSS_WEIGHT * F.smooth_l1_loss(
                aux['freq_recon'][aux['freq_mask']], aux['freq_target'][aux['freq_mask']])
        if future_target is not None:
            # Extract the future's low-pass trend and complementary residual.
            # This target split mirrors the time/frequency roles in MUSE.
            future_fft = torch.fft.rfft(future_target, dim=1)
            keep = min(DLF_FUTURE_LPF, future_fft.size(1))
            filtered_fft = torch.zeros_like(future_fft)
            filtered_fft[:, :keep, :] = future_fft[:, :keep, :]
            future_low = torch.fft.irfft(filtered_fft, n=future_target.size(1), dim=1)
            future_residual = future_target - future_low
            # Alpha is the validated domain prior.  Both paths stay in the
            # training graph, but the relevant domain receives most gradient.
            dual_loss = self.args.alpha * F.smooth_l1_loss(
                aux['time_future'], future_residual.detach()) + \
                (1.0 - self.args.alpha) * F.smooth_l1_loss(
                aux['freq_future'], future_low.detach())
            loss = loss + DLF_FUTURE_LOSS_WEIGHT * dual_loss
        return loss

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        # [NEW] Handle Tuple output for Latent Alignment
                        if isinstance(outputs, tuple) and len(outputs) == 4:
                            outputs, z_t_aligned, z_s_aligned, latent_aux = outputs
                        elif isinstance(outputs, tuple) and len(outputs) == 3:
                            outputs, z_t_aligned, z_s_aligned = outputs
                            latent_aux = None
                        else:
                            z_t_aligned, z_s_aligned, latent_aux = None, None, None

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss_pred = criterion(outputs, batch_y)
                        loss = loss_pred
                        loss = loss + self._latent_auxiliary_loss(latent_aux, batch_y)
                        # Normalised spectra prevent the large DC component
                        # from drowning out periodic frequencies at long horizons.
                        pred_spec = torch.fft.rfft(outputs, dim=1)
                        true_spec = torch.fft.rfft(batch_y, dim=1)
                        scale = true_spec.abs().mean(dim=1, keepdim=True).clamp_min(1e-4)
                        loss_spec = F.smooth_l1_loss(pred_spec.real / scale, true_spec.real / scale) + \
                                    F.smooth_l1_loss(pred_spec.imag / scale, true_spec.imag / scale)
                        loss = loss + DLF_FREQ_LOSS_WEIGHT * loss_spec

        
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    # [NEW] Handle Tuple output for Latent Alignment
                    if isinstance(outputs, tuple) and len(outputs) == 4:
                        outputs, z_t_aligned, z_s_aligned, latent_aux = outputs
                    elif isinstance(outputs, tuple) and len(outputs) == 3:
                        outputs, z_t_aligned, z_s_aligned = outputs
                        latent_aux = None
                    else:
                        z_t_aligned, z_s_aligned, latent_aux = None, None, None

                    f_dim = -1 if self.args.features == 'MS' else 0

                    pred_y = outputs[:, -self.args.pred_len:, f_dim:].to(self.device)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss_pred = criterion(pred_y, batch_y)
                    loss = loss_pred
                    loss = loss + self._latent_auxiliary_loss(latent_aux, batch_y)
                    pred_spec = torch.fft.rfft(pred_y, dim=1)
                    true_spec = torch.fft.rfft(batch_y, dim=1)
                    scale = true_spec.abs().mean(dim=1, keepdim=True).clamp_min(1e-4)
                    loss_spec = F.smooth_l1_loss(pred_spec.real / scale, true_spec.real / scale) + \
                                F.smooth_l1_loss(pred_spec.imag / scale, true_spec.imag / scale)
                    loss = loss + DLF_FREQ_LOSS_WEIGHT * loss_spec

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_gpu:
            self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))
        else:
            self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model',setting)
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location="cuda:0"))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
            # test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        # preds = np.array(preds)
        # trues = np.array(trues)
        # inputx = np.array(inputx)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)  # if there is not that line, ignore this

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{:.4f}, mae:{:.4f}, rse:{:.4f}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{:.4f}, mae:{:.4f}, rse:{:.4f}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
