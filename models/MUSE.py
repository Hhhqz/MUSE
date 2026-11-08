import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    
    LATENT_ALIGN_DIM = 16
    FREQ_MASK_RATIO = 0.25
   
    CPRC_RANK = 8
    PHASE_NUM_PERIODS = 4
    PHASE_TEMPERATURE = 0.2
    PHASE_RECENT_WEIGHT = 0.5
    PHASE_GATE_THRESHOLD = 0.5
    PHASE_GATE_TEMPERATURE = 0.1
   
    AMDF_PERIOD_MULTIPLIERS = (1, 2, 4)
    AMDF_RANK = 8

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.kernel = self.period_len
        self.lpf = configs.lpf
        self.alpha = configs.alpha
        self.dlf_enabled = bool(getattr(configs, "dlf_enabled", 1))
        self.cprc_enabled = bool(getattr(configs, "cprc_enabled", 0))
        self.cprc_rank = self.CPRC_RANK
        self.cprc_cap = float(getattr(configs, "cprc_cap", 0.05))
    
        self.amdf_enabled = bool(getattr(configs, "amdf_enabled", 0))
        self.phase_num_periods = self.PHASE_NUM_PERIODS
        self.phase_max_shift = max(0, int(getattr(configs, "phase_max_shift", 0)))
        self.phase_temperature = self.PHASE_TEMPERATURE
        self.phase_recent_weight = self.PHASE_RECENT_WEIGHT
        self.phase_gate_threshold = self.PHASE_GATE_THRESHOLD
        self.phase_gate_temperature = self.PHASE_GATE_TEMPERATURE

        self.seg_num_x = math.ceil(self.seq_len / self.period_len)
        self.seg_num_y = math.ceil(self.pred_len / self.period_len)
        self.padded_seq_len = self.seg_num_x * self.period_len
        self.sqrt_seg_num_x = math.ceil(math.sqrt(self.seq_len / self.period_len))
        self.sqrt_seg_num_y = math.ceil(math.sqrt(self.pred_len / self.period_len))

        self.TLinear1 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)
        self.TLinear2 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)
       
        self.conv1d = nn.Conv1d(1, 1, kernel_size=self.kernel + 1, stride=1,
                                padding=0, bias=False)
        self.conv_pad_left = self.kernel // 2
        self.conv_pad_right = self.kernel - self.conv_pad_left
        self.FLinear1 = nn.Linear(self.lpf, 2, bias=False).to(torch.cfloat)
        self.FLinear2 = nn.Linear(2, self.seg_num_y, bias=False).to(torch.cfloat)
        
        self.time_feature_dim = self.sqrt_seg_num_y ** 2
        default_latent = min(16, max(4, math.ceil(self.time_feature_dim / 2)))
        self.time_latent_dim = default_latent
        self.align_dim = self.LATENT_ALIGN_DIM
        self.time_mask_ratio = 0.0
        self.freq_mask_ratio = self.FREQ_MASK_RATIO
        self.shared_latent_mode = "alpha"

        self.time_encoder = nn.Sequential(
            nn.Linear(self.time_feature_dim, self.time_latent_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.time_latent_dim),
        )
        self.freq_encoder = nn.Sequential(
            nn.Linear(4, self.time_latent_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.time_latent_dim),
        )
        self.time_projector = nn.Sequential(
            nn.Linear(self.time_latent_dim, self.align_dim, bias=False),
            nn.LayerNorm(self.align_dim),
        )
        self.freq_projector = nn.Sequential(
            nn.Linear(self.time_latent_dim, self.align_dim, bias=False),
            nn.LayerNorm(self.align_dim),
        )
        
        self.time_reconstructor = nn.Linear(self.time_latent_dim, self.time_feature_dim, bias=False)
        self.freq_reconstructor = nn.Linear(self.time_latent_dim, 4, bias=False)
        self.time_shared = nn.Sequential(
            nn.Linear(self.time_latent_dim, self.align_dim, bias=False),
            nn.LayerNorm(self.align_dim),
        )
        self.freq_shared = nn.Sequential(
            nn.Linear(self.time_latent_dim, self.align_dim, bias=False),
            nn.LayerNorm(self.align_dim),
        )
        
        self.time_future_decoder = nn.Sequential(
            nn.Linear(self.align_dim, self.align_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.align_dim, self.pred_len, bias=False),
        )
        self.freq_future_decoder = nn.Sequential(
            nn.Linear(self.align_dim, self.align_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.align_dim, self.pred_len, bias=False),
        )

        self.periodic_encoder = nn.Sequential(
            nn.Linear(self.period_len, self.cprc_rank, bias=False),
            nn.GELU(),
        )
        self.periodic_decoder = nn.Linear(self.cprc_rank, self.pred_len, bias=False)
        nn.init.zeros_(self.periodic_decoder.weight)

        self.amdf_encoder = nn.Sequential(
            nn.Linear(self.period_len, self.AMDF_RANK, bias=False),
            nn.GELU(),
        )
        self.amdf_decoder = nn.Linear(self.AMDF_RANK, self.pred_len, bias=False)
        nn.init.zeros_(self.amdf_decoder.weight)

    @staticmethod
    def _masked_view(features, ratio):
        """Mask complete semantic tokens rather than scalar coordinates."""
        if ratio <= 0:
            return features, torch.zeros(features.shape[:-1], dtype=torch.bool, device=features.device)
        token_mask = torch.rand(features.shape[:-1], device=features.device) < ratio
        return features.masked_fill(token_mask.unsqueeze(-1), 0.0), token_mask

    def _phase_aligned_period(self, x, latest_period):
       
        available_periods = x.shape[-1] // self.period_len
        num_periods = min(self.phase_num_periods, available_periods)
        if num_periods < 2:
            return latest_period, latest_period.new_ones(latest_period.shape[:2])

        # [B, C, K - 1, P], from older to newer historical references.
        history = x[:, :, -(num_periods * self.period_len):-self.period_len]
        history = history.reshape(x.shape[0], x.shape[1], num_periods - 1, self.period_len)
        shifts = range(-self.phase_max_shift, self.phase_max_shift + 1)
        shifted_history = torch.stack(
            [torch.roll(history, shifts=shift, dims=-1) for shift in shifts], dim=-2
        )  # [B, C, K - 1, S, P]

        latest_shape = latest_period - latest_period.mean(dim=-1, keepdim=True)
        latest_shape = latest_shape / latest_shape.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        reference_shape = shifted_history - shifted_history.mean(dim=-1, keepdim=True)
        reference_shape = reference_shape / reference_shape.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-4)
        latest_norm = F.normalize(latest_shape, dim=-1, eps=1e-6).unsqueeze(-2).unsqueeze(-2)
        reference_norm = F.normalize(reference_shape, dim=-1, eps=1e-6)
        similarity = (latest_norm * reference_norm).sum(dim=-1)  # [B, C, K - 1, S]
        phase_weights = torch.softmax(similarity / self.phase_temperature, dim=-1)
        aligned_history = (phase_weights.unsqueeze(-1) * shifted_history).sum(dim=-2)

        period_similarity = (phase_weights * similarity).sum(dim=-1)
        period_weights = torch.softmax(period_similarity / self.phase_temperature, dim=-1)
        aligned_reference = (period_weights.unsqueeze(-1) * aligned_history).sum(dim=-2)
        match_confidence = (period_weights * period_similarity).sum(dim=-1)

        shift_count = similarity.shape[-1]
        if shift_count > 1:
            phase_entropy = -(phase_weights * phase_weights.clamp_min(1e-8).log()).sum(dim=-1)
            phase_concentration = 1.0 - phase_entropy / math.log(shift_count)
            phase_concentration = (period_weights * phase_concentration).sum(dim=-1)
        else:
            phase_concentration = torch.ones_like(match_confidence)
       
        match_reliability = 0.5 * (match_confidence + 1.0)
        confidence = match_reliability * (0.5 + 0.5 * phase_concentration)
        gate = torch.sigmoid(
            (confidence - self.phase_gate_threshold) / self.phase_gate_temperature
        )
        aligned_period = (self.phase_recent_weight * latest_period +
                          (1.0 - self.phase_recent_weight) * aligned_reference)
        return aligned_period, gate

    def _adaptive_multi_period_state(self, x, latest_period):
        
        states = []
        reliabilities = []
        for multiplier in self.AMDF_PERIOD_MULTIPLIERS:
            width = multiplier * self.period_len
            if x.shape[-1] < width:
                continue
            cycles = x[:, :, -width:].reshape(
                x.shape[0], x.shape[1], multiplier, self.period_len
            )
            current = cycles[:, :, -1]
            if multiplier > 1:
                template = cycles[:, :, :-1].mean(dim=-2)
                deviation = current - template
                error = (current - template).abs().mean(dim=-1)
                scale = current.abs().mean(dim=-1).clamp_min(1e-3)
                reliability = torch.exp(-error / scale).clamp_min(1e-3)
            else:
               
                aligned, phase_gate = self._phase_aligned_period(x, latest_period)
                deviation = aligned - latest_period
                reliability = phase_gate
            states.append(deviation)
            reliabilities.append(reliability)

        if not states:
            return latest_period.new_zeros(latest_period.shape), \
                latest_period.new_ones(latest_period.shape[:2])
        state = torch.stack(states, dim=-2)       # [B, C, scales, P]
        reliability = torch.stack(reliabilities, dim=-1)  # [B, C, scales]
        weights = reliability / reliability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        state = (state * weights.unsqueeze(-1)).sum(dim=-2)
        gate = reliability.mean(dim=-1).clamp(0.0, 1.0)
        return state, gate

    def forward(self, x):
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = (x - seq_mean).permute(0, 2, 1)
        x_raw = x
       
        latest_period = x[:, :, -self.period_len:]
        conv_input = F.pad(x.reshape(-1, 1, self.seq_len),
                           (self.conv_pad_left, self.conv_pad_right))
        x = self.conv1d(conv_input).reshape(-1, self.enc_in, self.seq_len) + x
        
        if self.padded_seq_len > self.seq_len:
            x = F.pad(x, (0, self.padded_seq_len - self.seq_len))
        x = x.reshape(batch_size, self.enc_in, self.seg_num_x,
                      self.period_len).permute(0, 1, 3, 2)

       
        x_o = F.pad(x, (0, self.sqrt_seg_num_x ** 2 - x.shape[-1]))
        x_o = x_o.reshape(batch_size, self.enc_in, self.period_len,
                          self.sqrt_seg_num_x, self.sqrt_seg_num_x)
        x_o = self.TLinear1(x_o).permute(0, 1, 2, 4, 3)
        x_t = self.TLinear2(x_o).permute(0, 1, 2, 4, 3)
        time_features = x_t.reshape(batch_size, self.enc_in, self.period_len, -1)
        x_t = time_features.permute(0, 1, 3, 2).reshape(
            batch_size, self.enc_in, -1).permute(0, 2, 1)

        
        x_fft = torch.fft.fft(x, dim=3)[:, :, :, :self.lpf]
        z_s = self.FLinear1(x_fft)
        pred_fft = self.FLinear2(z_s).reshape(batch_size, self.enc_in, self.period_len, -1)
        
        freq_features = torch.fft.ifft(pred_fft, dim=3).float()
        x_f = freq_features.permute(0, 1, 3, 2).reshape(
            batch_size, self.enc_in, -1).permute(0, 2, 1)

       
        out = x_t[:, :self.pred_len, :] * self.alpha + seq_mean + \
              x_f[:, :self.pred_len, :] * (1.0 - self.alpha)

       
        if not self.dlf_enabled:
            return out

       
        z_t = self.time_encoder(time_features)
        z_s_ri = torch.view_as_real(z_s).flatten(start_dim=-2).float()
        z_f = self.freq_encoder(z_s_ri)
       
        time_base = x_t[:, :self.pred_len, :]
        freq_base = x_f[:, :self.pred_len, :]
        out = self.alpha * time_base + (1.0 - self.alpha) * freq_base + seq_mean
        if self.cprc_enabled:
            periodic_input = latest_period
            periodic_gate = latest_period.new_ones(batch_size, self.enc_in)
            if self.amdf_enabled:
                periodic_input, periodic_gate = self._phase_aligned_period(x_raw, latest_period)
            periodic_code = self.periodic_encoder(periodic_input.reshape(-1, self.period_len))
            periodic_delta = self.periodic_decoder(periodic_code).reshape(
                batch_size, self.enc_in, self.pred_len).permute(0, 2, 1)
           
            out = out + self.cprc_cap * periodic_gate.unsqueeze(1) * torch.tanh(periodic_delta)

        if self.amdf_enabled:
           
            amdf_state, amdf_gate = self._adaptive_multi_period_state(
                x_raw, latest_period
            )
            amdf_code = self.amdf_encoder(amdf_state.reshape(-1, self.period_len))
            amdf_delta = self.amdf_decoder(amdf_code).reshape(
                batch_size, self.enc_in, self.pred_len
            ).permute(0, 2, 1)
            out = out + self.cprc_cap * amdf_gate.unsqueeze(1) * torch.tanh(amdf_delta)

        if self.training:
           
            time_masked, time_mask = self._masked_view(time_features, self.time_mask_ratio)
            freq_masked, freq_mask = self._masked_view(z_s_ri, self.freq_mask_ratio)
            z_t_masked = self.time_encoder(time_masked)
            z_f_masked = self.freq_encoder(freq_masked)
            time_shared = self.time_shared(z_t_masked)
            freq_shared = self.freq_shared(z_f_masked)
           
            time_summary = time_shared.mean(dim=-2)
            freq_summary = freq_shared.mean(dim=-2)
            if self.shared_latent_mode == "equal":
                shared_latent = 0.5 * (time_summary + freq_summary)
            else:
                shared_latent = self.alpha * time_summary + (1.0 - self.alpha) * freq_summary
            aux = {
                "time_recon": self.time_reconstructor(z_t_masked),
                "time_target": time_features.detach(),
                "time_mask": time_mask,
                "freq_recon": self.freq_reconstructor(z_f_masked),
                "freq_target": z_s_ri.detach(),
                "freq_mask": freq_mask,
                "time_shared": time_shared,
                "freq_shared": freq_shared,
               
                "shared_future": self.time_future_decoder(shared_latent).permute(0, 2, 1),
                "time_future": self.time_future_decoder(time_summary).permute(0, 2, 1),
                "freq_future": self.freq_future_decoder(freq_summary).permute(0, 2, 1),
            }
            return out, self.time_projector(z_t), self.freq_projector(z_f), aux
        return out
