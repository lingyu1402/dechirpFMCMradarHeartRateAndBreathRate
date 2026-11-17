"""
FMCW 生命体征 —— 真实工况版数值验证（与 fmcw_vitals_sim.m 同源）
================================================================================
（MATLAB 实跑心跳峰检不出后，逐层定位）：

【根因1 - 确定的建模错误】躯干原设 1.512m，胸腔 1.50m，仅差 12mm。
  距离分辨率 5cm（bin宽2.8cm）下两者落同一 bin，微动矢量在 bin 内相消，
  目标 bin 相位几乎检不到微动（呼吸卡17dB/心跳3dB）。
  修正：躯干移到 1.80m（与胸腔可分辨）。真实雷达里胸壁反射点自身在动、
  是其距离 bin 的主导散射点，身体其他部位分布在不同 bin——这才是正确建模。

【根因2 - 有害的处理步骤】back 参考的复数域体动对消
  ratio=zt·conj(zb)，在 7mm 体动（相位摆动~17rad）下 unwrap 出错、
  残留巨大，反而把心跳压到 5dB。体动是 0.05/0.11Hz 低频，与呼吸0.25/
  心跳1.1Hz 频率可分，靠"相噪对消 + 滑动中值去趋势"即可干净分离。
  修正：移除 back 复数对消，保留相噪对消（墙bin）+ 滑动中值去趋势。

【重要结论 - 相位与幅度解耦】躯干移开后，目标 bin 由胸腔主导，
  该 bin 相位 = 4πΔR/λ 只取决于位移、与回波幅度无关。
  故胸腔幅度 0.05 与 0.5 的检出结果几乎一致（心跳均 ~26dB）。
  此前"20:1 压缩导致心跳弱"是同 bin 抵消造成的假象，现已不成立。
  脚本保留两个幅度场景仅为展示这一解耦特性。

处理链（关键）：
  距离FFT -> 目标bin求和 -> 相噪对消(墙bin相位) -> 解缠相位->位移
  -> 滑动中值去趋势(去体动) -> 频谱(看呼吸/心跳频率)
"""
import numpy as np

# ============================ 参数 ============================
c=3e8; fc=60e9; lam=c/fc
B=3e9; Tsweep=0.1e-3; Tidle=49.9e-3; Tchirp=Tsweep+Tidle
slope=B/Tsweep; f0=fc-B/2
Nchirp=800; Rmax=3.0; R0=1.50
fb_max=2*slope*Rmax/c; Fs_adc=1.2*fb_max
Nfast=round(Fs_adc*Tsweep); Nfft_r=2**int(np.ceil(np.log2(Nfast)))
tfast=np.arange(Nfast)/Fs_adc
rangeAxis=np.arange(Nfft_r)*(Fs_adc/Nfft_r)*c/(2*slope)
tm=np.arange(Nchirp)*Tchirp; fs_slow=1/Tchirp
Nfft_s=2**16
f_ax=np.arange(Nfft_s)*fs_slow/Nfft_s

# 微动 / 体动 / 相噪
A_breath,f_breath=1.0e-3,0.25
A_heart,f_heart=0.25e-3,1.10
dR=A_breath*np.sin(2*np.pi*f_breath*tm)+A_heart*np.sin(2*np.pi*f_heart*tm+np.pi/3)
rng_body=np.random.default_rng(3)
body=(2.5e-3*np.sin(2*np.pi*0.05*tm)+1.5e-3*np.sin(2*np.pi*0.11*tm+0.7)
      +0.4e-3*np.cumsum(rng_body.standard_normal(Nchirp))/np.sqrt(Nchirp))

A_TORSO=1.0; A_CHEST_REAL=0.05; A_CHEST_DEMO=0.5
A_WALL=0.60; A_FURN=0.35
TORSO_R=1.80   # 修正：原1.512与胸腔同bin抵消，移到1.80m可分辨
# 散射点集合与 MATLAB 完全一致：chest / torso / wall / furn（无 back，因其不再参与对消）

def gen_phase_noise(N, fs, rms_deg=1.0, seed=7):
    rng=np.random.default_rng(seed)
    f=np.fft.rfftfreq(N,1/fs); ff=np.maximum(f,0.3)
    L_1hz=-25; L_floor=-70
    S=10**(L_1hz/10)/ff**2+10**(L_floor/10); S[0]=0
    H=np.sqrt(S*fs*N/2)
    Xf=H*(rng.standard_normal(len(f))+1j*rng.standard_normal(len(f)))
    pn=np.fft.irfft(Xf,N)
    return pn/pn.std()*np.deg2rad(rms_deg)

pn=gen_phase_noise(Nchirp,fs_slow,rms_deg=1.0)

# ============================ 回波仿真 ============================
def sim_echo(chest_amp, add_pn, add_body, noise_amp=0.001, noise_seed=11):
    b=body if add_body else np.zeros(Nchirp)
    pts={'chest':(R0+b+dR,chest_amp), 'torso':(TORSO_R+b,A_TORSO),
         'wall':(0.80+np.zeros(Nchirp),A_WALL),  # 墙静止（相噪参考）
         'furn':(2.30+np.zeros(Nchirp),A_FURN)}
    names=list(pts); Rarr=np.array([pts[n][0] for n in names]); aarr=np.array([pts[n][1] for n in names])
    win=0.5-0.5*np.cos(2*np.pi*np.arange(Nfast)/(Nfast-1))
    Xrng=np.zeros((Nchirp,Nfft_r),complex)
    rng=np.random.default_rng(noise_seed)
    for m in range(Nchirp):
        acc=np.zeros(Nfast,complex)
        for i in range(len(names)):
            tau=2*Rarr[i,m]/c
            acc+=aarr[i]*np.exp(1j*2*np.pi*(f0*tau+slope*tau*tfast-0.5*slope*tau**2))
        if noise_amp>0:
            acc+=noise_amp*(rng.standard_normal(Nfast)+1j*rng.standard_normal(Nfast))/np.sqrt(2)
        phase=pn[m] if add_pn else 0.0
        Xrng[m]=np.fft.fft(acc*win,Nfft_r)*np.exp(1j*phase)
    return Xrng

def bin_of(R): return int(round(2*slope*R/c/Fs_adc*Nfft_r))
bin_t,bin_wall=bin_of(R0),bin_of(0.80)
BIN_HW=1   # 目标bin现在由胸腔主导，无需宽窗接泄露；窄窗避免邻bin混叠

def zbin(Xrng,b0,hw=None):
    if hw is None: hw=BIN_HW
    s=slice(max(0,b0-hw),min(Nfft_r,b0+hw+1))
    return Xrng[:,s].sum(axis=1)

# ============================ 处理链 ============================
def moving_median_detrend(x, win_s=5.0):
    k=int(round(win_s/Tchirp))|1
    pad=np.pad(x,(k//2,k//2),mode='edge')
    trend=np.median(np.lib.stride_tricks.sliding_window_view(pad,k),axis=1)
    return x-trend

def process_vitals(Xrng):
    """相噪对消(墙bin) -> 解缠相位->位移 -> 滑动中值去趋势。
    不用 back 复数对消：已证实它在大体动下破坏信号，且体动与呼吸/心跳频率可分。"""
    zt,z_wall=zbin(Xrng,bin_t),zbin(Xrng,bin_wall)
    pn_est=np.angle(z_wall)-np.angle(z_wall).mean()
    ph=np.unwrap(np.angle(zt*np.exp(-1j*pn_est)))*lam/(4*np.pi)
    ph=ph-ph.mean()
    rb=moving_median_detrend(ph,5.0)
    dR_proc=rb-np.polyval(np.polyfit(tm,rb,1),tm)
    # 失败对照：不做相噪对消，直接解缠
    ph_fail=np.unwrap(np.angle(zt))*lam/(4*np.pi)
    ph_fail=ph_fail-ph_fail.mean()
    return dR_proc,ph_fail

def spectrum(x):
    s=np.abs(np.fft.fft(x*np.hanning(Nchirp),Nfft_s))
    m=s.max()
    if m<=0: m=1.0
    return 20*np.log10(s/m+1e-12)

def peak_sig(spec,ft,hw=0.05):
    pk=spec[(f_ax>ft-hw)&(f_ax<ft+hw)].max()
    base_b=(f_ax>0.05)&(f_ax<2.5)&(np.abs(f_ax-ft)>0.2)
    return pk-np.median(spec[base_b])

def peak_freq(spec,ft,hw=0.08):
    b=(f_ax>ft-hw)&(f_ax<ft+hw)
    return f_ax[b][np.argmax(spec[b])]

def peak_report(spec,label):
    fr,fh=peak_freq(spec,f_breath),peak_freq(spec,f_heart)
    sb,sh=peak_sig(spec,f_breath),peak_sig(spec,f_heart)
    print(f"  {label:<24} 呼吸 {fr:.3f}Hz ({sb:+.1f}dB)  心跳 {fh:.3f}Hz ({sh:+.1f}dB)  [真值 0.250/1.100]")
    return fr,fh

# ============================ 运行 ============================
print("="*74)
print("场景A 演示配置：胸腔0.5（信号主导），躯干1.8m，相噪+体动+噪声，完整链路")
Xa=sim_echo(A_CHEST_DEMO,True,True)
dRa,pf_a=process_vitals(Xa)
spec_a=spectrum(dRa); spec_af=spectrum(pf_a)
print("  [相噪对消前]"); peak_report(spec_af,"无相噪对消")
print("  [完整链路]"); peak_report(spec_a,"相噪对消+中值去体动")

print("\n场景B 真实比例：胸腔0.05（20:1），同样干扰与链路（与场景A对比相位-幅度解耦）")
Xb=sim_echo(A_CHEST_REAL,True,True)
dRb,pf_b=process_vitals(Xb)
spec_b=spectrum(dRb)
peak_report(spec_b,"真实0.05完整链路")

print("\n[多 realization 稳健性 · 场景A] 严格判据=峰高-局部基底(dB)，>5dB 显著")
scan=[(7,3,11),(7,3,2),(11,5,3),(20,10,7),(116,7,13),(3,11,29),
      (1,3,4),(9,6,8),(50,4,12),(77,9,19)]
pn0=pn.copy(); body0=body.copy()
sh=[];ss=[]
for pn_s,b_s,n_s in scan:
    pn[:]=gen_phase_noise(Nchirp,fs_slow,1.0,pn_s)
    rb=np.random.default_rng(b_s)
    body[:]=(2.5e-3*np.sin(2*np.pi*0.05*tm)+1.5e-3*np.sin(2*np.pi*0.11*tm+0.7)
             +0.4e-3*np.cumsum(rb.standard_normal(Nchirp))/np.sqrt(Nchirp))
    Xs=sim_echo(A_CHEST_DEMO,True,True,noise_amp=0.001,noise_seed=n_s)
    ds,_=process_vitals(Xs); sp=spectrum(ds)
    sh.append(peak_sig(sp,f_heart)); ss.append(peak_sig(sp,f_breath))
pn[:]=pn0; body[:]=body0
print(f"  心跳 中位{np.median(sh):.1f}dB 显著{sum(x>5 for x in sh)}/{len(sh)} | "
      f"呼吸 中位{np.median(ss):.1f}dB 显著{sum(x>5 for x in ss)}/{len(ss)}")

# ============================ 出图 ============================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(2,3,figsize=(15,8))
prof_a=np.abs(Xa.mean(axis=0)); prof_b=np.abs(Xb.mean(axis=0))
axes[0,0].plot(rangeAxis[:90],20*np.log10(prof_a[:90]/prof_a.max()+1e-9),'b')
axes[0,0].axvline(rangeAxis[bin_t],color='r',ls='--',label='胸腔(微动主导)')
axes[0,0].axvline(rangeAxis[bin_of(TORSO_R)],color='orange',ls=':',label='躯干')
axes[0,0].set_title('演示·距离像(躯干已移开)');axes[0,0].set_xlabel('距离 m');axes[0,0].legend(fontsize=7);axes[0,0].grid(True)
axes[0,1].plot(tm,dRa*1e3,'b',lw=1);axes[0,1].plot(tm,dR*1e3,'r--',lw=1)
axes[0,1].set_title('演示·慢时间位移(蓝=估计 红=真值)');axes[0,1].set_xlabel('时间 s');axes[0,1].grid(True)
axes[0,2].plot(f_ax,spec_a,'b');axes[0,2].plot(f_ax,spec_af,color='0.7',lw=0.8)
axes[0,2].axvline(f_breath,color='g',ls='--');axes[0,2].axvline(f_heart,color='m',ls='--')
axes[0,2].set_xlim(0,3);axes[0,2].set_ylim(-80,0)
axes[0,2].set_title('演示·频谱(蓝=对消后 灰=未对消)');axes[0,2].set_xlabel('频率 Hz');axes[0,2].grid(True)
prof_b=np.abs(Xb.mean(axis=0))
axes[1,0].plot(rangeAxis[:90],20*np.log10(prof_b[:90]/prof_b.max()+1e-9),'b')
axes[1,0].axvline(rangeAxis[bin_t],color='r',ls='--');axes[1,0].set_title('真实0.05·距离像');axes[1,0].set_xlabel('距离 m');axes[1,0].grid(True)
axes[1,1].plot(tm,dRb*1e3,'b',lw=1);axes[1,1].plot(tm,dR*1e3,'r--',lw=1)
axes[1,1].set_title('真实0.05·位移(与0.5几乎一致)');axes[1,1].set_xlabel('时间 s');axes[1,1].grid(True)
axes[1,2].plot(f_ax,spec_b,'b');axes[1,2].axvline(f_breath,color='g',ls='--');axes[1,2].axvline(f_heart,color='m',ls='--')
axes[1,2].set_xlim(0,3);axes[1,2].set_ylim(-80,0)
axes[1,2].set_title('真实0.05·频谱(双峰清晰)');axes[1,2].set_xlabel('频率 Hz');axes[1,2].grid(True)
plt.tight_layout()
out='./fmcw_vitals_realworld_compare.png'
plt.savefig(out,dpi=130)
print(f"\n对比图已保存: {out}")
