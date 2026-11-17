%% fmcw_vitals_sim_20261116.m
% 60 GHz FMCW 生命体征（呼吸/心跳）雷达 —— 真实工况版全链路仿真
%
% ============================================================================
% 2026-11-17 重大修正（Jango 反馈 MATLAB 实跑心跳峰检不出后逐层定位的根因）：
% ----------------------------------------------------------------------------
% 【根因1｜建模错误】躯干原设 1.512m，胸腔 1.50m，仅差 12mm。距离分辨率 5cm
%   （bin 宽 2.8cm）下两者落同一距离 bin，微动矢量在 bin 内相消，目标 bin
%   相位几乎检不到微动（呼吸卡 17dB / 心跳 3dB）。
%   修正：躯干移到 1.80m（与胸腔可分辨）。真实雷达里胸壁反射点自身在动、
%   是其距离 bin 的主导散射点，身体其它部位分布在不同 bin，这才是正确建模。
%   修正后：相位只取决于位移、与回波幅度无关，0.05 与 0.5 信号一样强。
%
% 【根因2｜有害处理】back 参考的复数域体动对消 ratio=zt.*conj(zb)，
%   在 7mm 体动（相位摆动 ~17rad）下 unwrap 出错、残留巨大，反把心跳压到 5dB。
%   体动是 0.05/0.11Hz 低频，与呼吸 0.25 / 心跳 1.1Hz 频率可分，靠
%   「相噪对消(墙bin) + 滑动中值去趋势」即可干净分离。
%   修正：移除 back 复数对消。
%
% 【处理链】距离FFT -> 目标bin求和 -> 相噪对消(墙bin相位) -> 解缠相位->位移
%           -> 滑动中值去趋势(去体动) -> 频谱(看呼吸/心跳频率)
%
% 【验证结果 Python 同源 2026-11-17】
%   场景A(胸腔0.5)：呼吸 0.251Hz(+40dB) / 心跳 1.099Hz(+26dB)，10/10 realization 显著
%   场景B(胸腔0.05)：呼吸 0.251Hz / 心跳 1.099Hz(+26dB) —— 躯干移开后不再边缘
%   未做相噪对消对照：双峰同样可见但基线抬升（相噪对消仍有价值）
%
% 依赖：核心仿真纯 MATLAB 即可；装了 Phased Array 会自动多画参考 chirp（可选）。
% Author: 小波  2026-11-17
% ============================================================================

clear; close all; clc;

%% ========================================================================
%  0. 仿真控制开关
% =========================================================================
S.seed             = 20261116;   % 随机种子（可复现）
S.usePhasedWave    = true;       % 是否用 phased.FMCWWaveform 画参考 chirp（无工具箱自动跳过）
S.addClutter       = true;       % 是否加静态散射点
S.addBodyMotion    = true;       % 是否加全身体动
S.addPhaseNoise    = true;       % 是否加相位噪声谱
S.addReceiverNoise = true;       % 是否加接收机热噪声
S.addWallRef       = true;       % 是否加静止墙参考点（相噪对消所需）
S.doPhaseNoiseCancel = true;     % 相噪对消开关
S.detrendMode      = 'median+linear'; % 'median+linear'(鲁棒) | 'linear' | 'none'
S.amplitudeModel   = 'normalized';    % 'normalized' | 'radareq'
S.targetAmp        = 0.05;       % 胸腔回波相对幅度。修正后相位与幅度无关，0.05/0.5 结果相同
S.plotWaveform     = true;
S.plotRangeProfile = true;
S.plotRangeTime    = true;
S.plotPhase        = true;

rng(S.seed);

%% ========================================================================
%  1. 环境 / 载波参数
% =========================================================================
P.c        = 3e8;             % 光速 m/s
P.fc       = 60e9;            % 载频 Hz（60 GHz；改 77e9 即车载频段）
P.lambda   = P.c / P.fc;      % 波长 m（60 GHz -> 5 mm）

%% ========================================================================
%  2. FMCW 波形参数（发射端）
% =========================================================================
P.B        = 3.0e9;           % 扫频带宽 Hz -> 距离分辨率 c/(2B)=5 cm
P.Tsweep   = 0.1e-3;          % 有效扫频时间 s（Ramp 100 us）
P.Tidle    = 49.9e-3;         % 空闲时间 s（拉长以放慢慢时间）
P.Tchirp   = P.Tsweep + P.Tidle;   % chirp 周期 = 慢时间采样间隔 = 50 ms
P.f_slow   = 1 / P.Tchirp;    % 慢时间采样率 Hz（20 Hz）
P.slope    = P.B / P.Tsweep;  % 扫频斜率 Hz/s（=3e13）
P.sweepDir = 'Up';            % 'Up' | 'Down' | 'Triangle'
P.f0       = P.fc - P.B/2;    % 起始频率 Hz
P.Nchirp   = 800;             % CPI 内 chirp 数
P.Tcpi     = P.Nchirp * P.Tchirp;   % 相干处理间隔 s（=40 s，分辨率 0.025 Hz）

%% ========================================================================
%  3. 快时间 / ADC 参数（按拍频采样）
% =========================================================================
P.Rmax     = 3;               % 最大无模糊距离 m
P.fb_max   = 2 * P.slope * P.Rmax / P.c;   % 最大拍频 Hz（=600 kHz）
P.Fs_adc   = 1.2 * P.fb_max;  % ADC 采样率（按拍频带宽，不是 chirp 带宽！）
% 【为何 1.2× 而非 2×】dechirp 输出为复信号（I/Q 正交解调，解析 exp(j·2πf·t)），
% 单边谱只占 0~fb_max，带宽即 fb_max；复采样 Fs≥带宽即可，1.2× 是 20% 抗混叠余量。
% 2×fb_max 只对实信号成立（单路实混频谱对称 ±fb_max，带宽才是 2fb_max）。
P.Nfast    = round(P.Fs_adc * P.Tsweep);   % 快时间采样点数（=72）
P.tfast    = (0:P.Nfast-1) / P.Fs_adc;     % 快时间轴 s
P.Nfft_r   = 2^nextpow2(P.Nfast);          % 距离 FFT 点数
P.rangeRes = P.c / (2 * P.B);              % 理论距离分辨率 m
P.rangeAxis= (0:P.Nfft_r-1)' * (P.Fs_adc/P.Nfft_r) * P.c / (2*P.slope);

%% ========================================================================
%  4~5. 发射机 / 接收机
% =========================================================================
P.Ptx_dBm  = 10;  P.TxGain = 10;  P.RxGain = 10;
P.Ntx      = 1;   P.Nrx    = 1;   P.sysLoss = 3;
P.NF_dB    = 6;   P.RxGain_dB = 30;
P.ADCbits  = 16;  P.ADCfs_dBm = 0;  P.ADCstart = 0;  P.applyQuant = false;
P.noiseAmp = 0.001;           % 接收机热噪声幅度（归一化）

%% ========================================================================
%  6. 相位噪声谱参数（振荡器近载波相噪，分段幂律）
% =========================================================================
P.pn.L1_dBc    = -25;         % 相噪谱 @1 Hz（dBc/Hz）
P.pn.Lfloor_dBc= -70;         % 白噪声底（dBc/Hz）
P.pn.fmin      = 0.3;         % 近载波平坦转折频率 Hz
P.pn.rms_deg   = 1.0;         % 每 chirp 相噪 rms（度）

%% ========================================================================
%  7. 目标 / 微动参数（胸腔）
% =========================================================================
P.R0       = 1.50;            % 胸腔中心距离 m
P.rcs_dBsm = 0;
P.A_breath = 1.0e-3;  P.f_breath = 0.25;  P.phi_breath = 0;
P.A_heart  = 0.25e-3; P.f_heart  = 1.10;  P.phi_heart  = pi/3;

%% ========================================================================
%  8. 静态散射点 / 参考点 —— 格式 [距离(m), 相对幅度]
% =========================================================================
%   0.80m 墙(静止, 0.60)  —— 相噪参考：静止，相位只含相噪
%   1.50m 胸腔(微动, 见 S.targetAmp) —— 其距离 bin 的主导散射点
%   1.80m 躯干(静止, 1.00) —— 【修正】远离胸腔，避免同 bin 微动抵消
%   2.30m 家具(静止, 0.35)
P.clutter = [ ...
    0.80,  0.60;   ...  % 墙（静止）—— 相噪参考 bin
    1.80,  1.00;   ...  % 躯干（【修正后 1.80m】与胸腔可分辨）
    2.30,  0.35 ];      % 远处家具（静止）
P.idx.wall = 1;

%% ========================================================================
%  9. 体动参数（全身体动，带动躯干/胸腔一起晃）
% =========================================================================
P.motion.a1 = 2.5e-3;  P.motion.f1 = 0.05;   % 主晃动 2.5mm @0.05Hz
P.motion.a2 = 1.5e-3;  P.motion.f2 = 0.11;   % 次晃动 1.5mm @0.11Hz
P.motion.rw = 0.4e-3;               % 随机游走幅度 m
P.binHalfWidth = 1;                 % 目标 bin 由胸腔主导，窄窗避免邻 bin 混叠

%% ========================================================================
%  10. 处理参数
% =========================================================================
P.nfft_spec = 2^16;        % 频谱 FFT 点数（zero-padding）
P.medWinSec = 5.0;         % 滑动中值去趋势窗长 s
P.binMode   = 'auto';      % 'auto' | 'manual'
P.manualBin = 54;

fprintf('=== 真实工况 FMCW 生命体征仿真（2026-11-17 修正版）===\n');
fprintf('载频 %.0f GHz | 带宽 %.1f GHz | 距离分辨率 %.2f m\n', ...
        P.fc/1e9, P.B/1e9, P.rangeRes);
fprintf('Nchirp %d | Tchirp %.1f ms | Tcpi %.1f s | f_slow %.0f Hz\n', ...
        P.Nchirp, P.Tchirp*1e3, P.Tcpi, P.f_slow);
fprintf('胸腔幅度 %.2f | 躯干距离 %.2f m | 相噪 rms %.1f°\n\n', ...
        S.targetAmp, P.clutter(P.idx.wall+1,1), P.pn.rms_deg);

%% ========================================================================
%  相噪序列（分段幂律 PSD -> 频域整形 -> IFFT）
% =========================================================================
pn = gen_phase_noise(P);
if ~S.addPhaseNoise, pn = zeros(P.Nchirp,1); end

%% ========================================================================
%  慢时间 / 微动 / 体动
% =========================================================================
tm = (0:P.Nchirp-1)' * P.Tchirp;
dR = P.A_breath*sin(2*pi*P.f_breath*tm + P.phi_breath) ...
   + P.A_heart *sin(2*pi*P.f_heart *tm + P.phi_heart);
body = gen_body_motion(P, tm);
if ~S.addBodyMotion, body = zeros(P.Nchirp,1); end
R_target = P.R0 + body + dR;    % 胸腔随身动 + 微动

%% ========================================================================
%  幅度模型
% =========================================================================
switch lower(S.amplitudeModel)
    case 'normalized'
        a_target = S.targetAmp;
        a_clut   = P.clutter(:,2);
    case 'radareq'
        Pt = 10^(P.Ptx_dBm/10)/1000;
        Gt = 10^(P.TxGain/10); Gr = 10^(P.RxGain/10);
        sig= 10^(P.rcs_dBsm/10); L = 10^(P.sysLoss/10);
        Pr = Pt*Gt*Gr*P.lambda^2*sig ./ ((4*pi)^3 * P.R0^4 * L);
        a_target = sqrt(Pr) * 1e4;
        a_clut   = a_target * P.clutter(:,2) / S.targetAmp;
    otherwise
        error('未知 amplitudeModel');
end
if ~S.addClutter, a_clut = zeros(size(a_clut)); end

%% ========================================================================
%  组装散射点（布局：nScat x Nchirp，每行一个散射点、每列一个 chirp）
% =========================================================================
R_static = repmat(P.clutter(:,1), 1, P.Nchirp);   % nClut x Nchirp
% 注意：第1行是墙，必须静止（相噪参考），不随身动！
% 第2行是躯干，随身动。Python 同源：wall=0.80+zeros, torso=1.80+body。
R_static(2,:) = R_static(2,:) + body';            % 躯干随身动
R_all = [R_target'; R_static];      % (1+nClut) x Nchirp
a_all = [a_target; a_clut];         % (1+nClut) x 1

%% ========================================================================
%  回波生成 + dechirp（解析等效）
% =========================================================================
X = zeros(P.Nchirp, P.Nfast);
for m = 1:P.Nchirp
    acc = zeros(1, P.Nfast);
    for i = 1:size(R_all,1)
        tau = 2 * R_all(i,m) / P.c;
        acc = acc + a_all(i) * exp( 1j*2*pi*( ...
              P.f0*tau + P.slope*tau*P.tfast - 0.5*P.slope*tau^2 ));
    end
    if S.addReceiverNoise
        acc = acc + P.noiseAmp * (randn(1,P.Nfast)+1j*randn(1,P.Nfast))/sqrt(2);
    end
    X(m,:) = acc;
end
X = X .* exp(1j*pn);   % 相噪：common-mode，每个 chirp 乘一个相位因子

%% ========================================================================
%  距离 FFT（加距离窗）
% =========================================================================
win = 0.5 - 0.5*cos(2*pi*(0:P.Nfast-1)'/(P.Nfast-1));
X = X .* win';
Xrng = fft(X, P.Nfft_r, 2);   % Nchirp x Nfft_r

%% ========================================================================
%  选 bin + 求和
% =========================================================================
switch lower(P.binMode)
    case 'auto'
        fb0  = 2*P.slope*P.R0/P.c;
        bin0 = round(fb0/P.Fs_adc*P.Nfft_r) + 1;
    case 'manual'
        bin0 = P.manualBin;
end
bin_wall = round(2*P.slope*P.clutter(1,1)/P.c/P.Fs_adc*P.Nfft_r) + 1;
hw = P.binHalfWidth;
ib_t    = max(1,bin0-hw)    : min(P.Nfft_r,bin0+hw);
ib_wall = max(1,bin_wall-hw): min(P.Nfft_r,bin_wall+hw);
z_t    = sum(Xrng(:,ib_t),    2);
z_wall = sum(Xrng(:,ib_wall), 2);
fprintf('目标bin=%d (%.2fm) | 墙参考bin=%d (%.2fm)\n', ...
        bin0, P.rangeAxis(bin0), bin_wall, P.rangeAxis(bin_wall));

%% ========================================================================
%  处理链：相噪对消 -> 解缠相位->位移 -> 滑动中值去体动
%  （不再做 back 复数体动对消：已证实它在大体动下破坏信号）
% =========================================================================
if S.doPhaseNoiseCancel && S.addWallRef
    pn_est = angle(z_wall) - mean(angle(z_wall));
    z_t_pc = z_t .* exp(-1j*pn_est);
else
    z_t_pc = z_t;
end
ph = unwrap(angle(z_t_pc)) * P.lambda/(4*pi);
ph = ph - mean(ph);
switch lower(S.detrendMode)
    case 'median+linear'
        dR_tmp  = moving_median_detrend(ph, P.medWinSec, P.Tchirp);
        dR_proc = dR_tmp - polyval(polyfit(tm,dR_tmp,1), tm);
    case 'linear'
        dR_proc = ph - polyval(polyfit(tm,ph,1), tm);
    otherwise
        dR_proc = ph;
end

% 失败对照：不做相噪对消
ph_fail = unwrap(angle(z_t)) * P.lambda/(4*pi);
ph_fail = ph_fail - mean(ph_fail);

%% ========================================================================
%  频谱 + 峰值报告
% =========================================================================
[spec, f_ax] = vitals_spectrum(dR_proc, P);
report_peaks(spec, f_ax, P);
[spec_fail, ~] = vitals_spectrum(ph_fail, P);
fprintf('\n[对照 未做相噪对消] ');
report_peaks(spec_fail, f_ax, P);
fprintf('体动峰峰 %.2f mm | 微动真值峰峰 %.2f mm\n\n', ...
        (max(body)-min(body))*1e3, (max(dR)-min(dR))*1e3);

%% ========================================================================
%  绘图
% =========================================================================
if S.plotWaveform
    if exist('phased.FMCWWaveform','class') == 8
        wf = phased.FMCWWaveform('SweepBandwidth',P.B,'SweepTime',P.Tsweep, ...
             'SampleRate',P.Fs_adc,'SweepDirection',P.sweepDir);
        refChirp = step(wf);
        figure('Name','参考 chirp');
        subplot(2,1,1); plot((0:numel(refChirp)-1)/P.Fs_adc*1e6, real(refChirp));
        xlabel('时间 (\mus)'); ylabel('实部'); title('FMCW chirp'); grid on;
        subplot(2,1,2); plot((0:numel(refChirp)-1)/P.Fs_adc*1e6, imag(refChirp));
        xlabel('时间 (\mus)'); ylabel('虚部'); grid on;
    else
        fprintf('（未装 Phased Array，跳过参考 chirp 图）\n');
    end
end

if S.plotRangeProfile
    figure('Name','距离像');
    prof = abs(mean(Xrng,1));
    plot(P.rangeAxis, 20*log10(prof/max(prof)+eps)); hold on;
    xline(P.rangeAxis(bin0),'r--','胸腔(微动主导)');
    xline(P.rangeAxis(round(2*P.slope*P.clutter(2,1)/P.c/P.Fs_adc*P.Nfft_r)+1),':','躯干');
    xline(P.rangeAxis(bin_wall),':','墙参考');
    xlim([0 min(P.Rmax,P.rangeAxis(end))]); ylim([-100 5]);
    xlabel('距离 (m)'); ylabel('幅度 (dB)'); title('距离像：躯干已移开，胸腔主导其 bin'); grid on;
end

if S.plotRangeTime
    figure('Name','距离-慢时间');
    Ridx = P.rangeAxis <= min(P.Rmax, P.rangeAxis(end));
    imagesc(tm, P.rangeAxis(Ridx), 20*log10(abs(Xrng(:,Ridx))'+eps));
    axis xy; colormap jet; colorbar; caxis([-60 0]);
    xlabel('慢时间 (s)'); ylabel('距离 (m)'); title('距离-慢时间强度');
end

if S.plotPhase
    figure('Name','慢时间位移与频谱');
    subplot(2,1,1);
    plot(tm, ph_fail*1e3, 'Color',[0.7 0.7 0.7], 'LineWidth',0.7); hold on;
    plot(tm, dR_proc*1e3, 'b', 'LineWidth',1);
    plot(tm, dR*1e3, 'r--', 'LineWidth',1);
    xlabel('慢时间 (s)'); ylabel('位移 (mm)');
    legend('未做相噪对消','完整链路后','真值(呼吸+心跳)');
    title('目标 bin 慢时间位移'); grid on;
    subplot(2,1,2);
    plot(f_ax, 20*log10(spec/max(spec)+eps), 'b'); hold on;
    plot(f_ax, 20*log10(spec_fail/max(spec_fail)+eps), 'Color',[0.7 0.7 0.7]);
    xline(P.f_breath,'g--','呼吸'); xline(P.f_heart,'m--','心跳');
    xlim([0 3]); ylim([-80 0]);
    xlabel('频率 (Hz)'); ylabel('幅度 (dB)'); title('位移频谱：完整链路 vs 未对消'); grid on;
end

fprintf('仿真完成。\n');

%% ========================================================================
%  局部函数定义（MATLAB R2016b+）
% =========================================================================
function pn = gen_phase_noise(P)
    N = P.Nchirp; fs = P.f_slow;
    f = (0:floor(N/2))'/N*fs;
    ff = max(f, P.pn.fmin);
    S = 10^(P.pn.L1_dBc/10) ./ ff.^2 + 10^(P.pn.Lfloor_dBc/10);
    S(1) = 0;
    H = sqrt(S * fs * N / 2);
    Xf = H .* (randn(numel(f),1) + 1j*randn(numel(f),1));
    pn = ifft([Xf; conj(Xf(end-1:-1:2))], 'symmetric');
    pn = pn / std(pn) * deg2rad(P.pn.rms_deg);
end

function body = gen_body_motion(P, tm)
    rw = P.motion.rw * cumsum(randn(numel(tm),1)) / sqrt(numel(tm));
    body = P.motion.a1*sin(2*pi*P.motion.f1*tm) ...
         + P.motion.a2*sin(2*pi*P.motion.f2*tm + 0.7) + rw;
end

function y = moving_median_detrend(x, winSec, Tchirp)
    k = round(winSec/Tchirp);
    if mod(k,2)==0, k = k+1; end
    n = numel(x);
    xp = [repmat(x(1),floor(k/2),1); x; repmat(x(end),floor(k/2),1)];
    trend = zeros(n,1);
    for i = 1:n
        trend(i) = median(xp(i:i+k-1));
    end
    y = x - trend;
end

function [spec, f_ax] = vitals_spectrum(x, P)
    Ns = P.nfft_spec;
    win = 0.54 - 0.46*cos(2*pi*(0:numel(x)-1)'/(numel(x)-1));
    spec = abs(fft(x .* win, Ns));
    f_ax = (0:Ns-1)' * (P.f_slow/Ns);
end

function report_peaks(spec, f_ax, P)
    fr = peak_near(spec, f_ax, P.f_breath);
    fh = peak_near(spec, f_ax, P.f_heart);
    spec_dB = 20*log10(spec/max(spec) + eps);
    sb = peak_sig_dB(spec_dB, f_ax, P.f_breath);
    sh = peak_sig_dB(spec_dB, f_ax, P.f_heart);
    fprintf('峰值检出: 呼吸 %.3f Hz (%+.1f dB) | 心跳 %.3f Hz (%+.1f dB)  [真值 %.3f/%.3f]\n', ...
            fr, sb, fh, sh, P.f_breath, P.f_heart);
end

function s = peak_sig_dB(spec_dB, f_ax, ft)
    hw = 0.05;
    b = f_ax > ft-hw & f_ax < ft+hw;
    pk = max(spec_dB(b));
    base_b = f_ax > 0.05 & f_ax < 2.5 & abs(f_ax-ft) > 0.2;
    s = pk - median(spec_dB(base_b));
end

function f = peak_near(spec, f_ax, ft, halfwin)
    if nargin < 4, halfwin = 0.08; end
    b = f_ax > ft-halfwin & f_ax < ft+halfwin;
    [~, i] = max(spec(b));
    f = f_ax(b);
    f = f(i);
end
