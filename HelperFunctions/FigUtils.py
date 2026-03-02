import numpy as np

import matplotlib
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FixedFormatter, LogFormatterSciNotation
from matplotlib import pyplot as plt


import scipy.linalg as linalg
from scipy.fft import fft, fftfreq

from scipy.signal import welch, get_window
from scipy.signal import periodogram, get_window

from tudatpy.astro.time_conversion import DateTime
#from tudatpy.numerical_simulation import Time

from datetime import datetime,timedelta
from pathlib import Path



def ConvertToDateTime(float_time_J2000):
    time_DateTime = []

    for t in float_time_J2000:
        t = DateTime.from_epoch(t).to_python_datetime()
        time_DateTime.append(t)

    return time_DateTime


# ##############################################################################################
# # RESIDUALS RSW
# ##############################################################################################
def Residuals_RSW(residuals_RSW, time,type="normal",title=None):

    time_dt = ConvertToDateTime(time)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    if type == "normal":
        labels = [r'$\Delta R$', r'$\Delta S$', r'$\Delta W$']
    elif type == "difference":
        labels = [r'$Difference \Delta R$', r'$Difference \Delta S$', r'$Difference \Delta W$']

    Y = [residuals_RSW[:,0], residuals_RSW[:,1], residuals_RSW[:,2]]

    # Pick consistent colors for R,S,W from the current color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = (colors + colors + colors)[:3]  # ensure at least 3

    for k, ax in enumerate(axes):
        # Plot all three series; make the current one opaque, others faint
        for j in range(3):
            alpha = 1.0 if j == k else 0.50   # 50% opacity for the non-focus series
            lw    = 1.6 if j == k else 1.0
            z     = 3 if j == k else 1
            ax.plot(time_dt, Y[j], color=colors[j], alpha=alpha, lw=lw, zorder=z)
            if j==k: 
                m = np.nanmax(np.abs(Y[j]))
                ax.set_ylim(-1.1*m, 1.1*m)

                #ax.set_ylim(-1.1*min(Y[j]), 1.1*max(Y[j]))
        
        ax.set_ylabel(f'{labels[k]} [km]')
        ax.grid(True, alpha=0.3)
        

    axes[-1].set_xlabel('Time [Date format]')

    # --- mdates: nice date ticks/labels on the shared x-axis ---
    locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
    formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    # (optional) minor ticks if you want extra granularity:
    # axes[-1].xaxis.set_minor_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
    if type == "normal":
        fig.suptitle('Residuals in RSW Frame') #, y=0.995)
    
    elif type == "difference":
        fig.suptitle('Difference in Residuals RSW') #, y=0.995)
    if title != None:
        fig.suptitle(title)

    return fig

# ##############################################################################################
# # COMPONENT PLOTS
# ##############################################################################################
def Plot_Components(array_Y, time,labels=None):

    time_dt = ConvertToDateTime(time)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    if labels == None:
        labels = [r'$\Delta X$', r'$\Delta Y$', r'$\Delta Z$']

    Y = [array_Y[:,0], array_Y[:,1], array_Y[:,2]]

    # Pick consistent colors for R,S,W from the current color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = (colors + colors + colors)[:3]  # ensure at least 3

    for k, ax in enumerate(axes):
        # Plot all three series; make the current one opaque, others faint
        for j in range(3):
            alpha = 1.0 if j == k else 0.50   # 50% opacity for the non-focus series
            lw    = 1.6 if j == k else 1.0
            z     = 3 if j == k else 1
            ax.plot(time_dt, Y[j], color=colors[j], alpha=alpha, lw=lw, zorder=z)
            if j==k: 
                m = np.nanmax(np.abs(Y[j]))
                ax.set_ylim(-1.1*m, 1.1*m)

                #ax.set_ylim(-1.1*min(Y[j]), 1.1*max(Y[j]))
        
        ax.set_ylabel(f'{labels[k]} [km]')
        ax.grid(True, alpha=0.3)
        

    axes[-1].set_xlabel('Time [Date format]')

    # --- mdates: nice date ticks/labels on the shared x-axis ---
    locator   = mdates.AutoDateLocator()               # chooses sensible tick spacing
    formatter = mdates.ConciseDateFormatter(locator)   # compact, smart formatting
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    # (optional) minor ticks if you want extra granularity:
    # axes[-1].xaxis.set_minor_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
    #fig.suptitle('Residuals in RSW Frame') #, y=0.995)
    
    # elif type == "difference":
    #     fig.suptitle('Difference in Residuals RSW Neptune and SSB') #, y=0.995)
    

    return fig


def Residuals_Magnitude_Compare(residuals, time,
                                fig=None, ax=None,
                                label_prefix=None,
                                line_width=1.6,
                                overlay_idx=0):
    """
    Plot magnitude of residual vectors over time, supporting overlays.
    residuals : ndarray [N x 3] or [N x m] -> will compute norm across axis=1
    time      : ndarray of times (float or datetime-like)
    """

    time_dt = ConvertToDateTime(time)

    created = False
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        created = True

    # Compute magnitude
    mag = np.linalg.norm(residuals, axis=1)

    # ---- NEW: color/linestyle per overlay
    colors = _overlay_colors(overlay_idx)
    linestyles = ['solid', '--', '-.', ':']
    ls = linestyles[overlay_idx % len(linestyles)]

    # Label
    label = f"{label_prefix} |Δ|" if label_prefix else "|Δ|"

    ax.plot(time_dt, mag,
            color=colors[0], linestyle=ls,
            lw=line_width, label=label)

    ax.set_ylabel('difference between 2 step sizes [m]')
    #ax.set_yscale("log")   # <-- log scale on Y axis
    ax.grid(True, alpha=0.3)

    if created:
        ax.set_xlabel('Time')
        locator   = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.suptitle('Residual Magnitude')

    ax.legend(loc='lower right')

    return fig, ax

def Plot_Polynomial_Fit(m,b,time,fig,ax,label):
    
    time_dt = ConvertToDateTime(time)
    ax.plot(time_dt, m*time + b, "--", label=f"{label} trend")

    return fig,ax

# ##############################################################################################
# # RESIDUALS RSW can compare
# ##############################################################################################
# def Residuals_RSW_Compare(residuals_RSW, time,
#                        fig=None, axes=None,
#                        label_prefix=None,
#                        faint_alpha=0.50,
#                        line_width_main=1.6,
#                        line_width_other=1.0,
#                        set_panel_limits=True,
#                        expand_only=True):
#     """
#     Overlay-friendly RSW residual plotter.

#     Parameters
#     ----------
#     residuals_RSW : (N,3) array
#         Columns: [ΔR, ΔS, ΔW] in km (or your units).
#     time : (N,) array-like
#         Epochs convertible by ConvertToDateTime(...).
#     fig, axes : optional
#         If provided, must be (Figure, ndarray/list of 3 Axes). New lines are drawn on them.
#         If None, a fresh 3x1 figure is created and returned.
#     label_prefix : str or None
#         If given, labels become f"{label_prefix} ΔR/S/W" on the focused row legend.
#     faint_alpha : float
#         Opacity for the two non-focused components.
#     set_panel_limits : bool
#         If True, set y-limits from THIS dataset’s focused component per panel.
#     expand_only : bool
#         If True and set_panel_limits is True, only expand current limits (don’t shrink).

#     Returns
#     -------
#     fig, axes
#     """
#     time_dt = ConvertToDateTime(time)

#     created = False
#     if fig is None or axes is None:
#         fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
#         created = True

#     labels = [r'$\Delta R$', r'$\Delta S$', r'$\Delta W$']
#     Y = [residuals_RSW[:, 0], residuals_RSW[:, 1], residuals_RSW[:, 2]]

#     # colors: consistent for R, S, W
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     colors = (colors + colors + colors)[:3]

#     for k, ax in enumerate(axes):
#         # Plot all three series; make the current one opaque, others faint
#         for j in range(3):
#             alpha = 1.0 if j == k else faint_alpha
#             lw    = line_width_main if j == k else line_width_other
#             z     = 3 if j == k else 1
#             lab   = (f"{label_prefix} {labels[j]}" if label_prefix else f"{labels[j]}") if j == k else None
#             ax.plot(time_dt, Y[j], color=colors[j], alpha=alpha, lw=lw, zorder=z, label=lab)

#         ax.set_ylabel(f'{labels[k]} [km]')
#         ax.grid(True, alpha=0.3)

#         if set_panel_limits:
#             y_main = np.asarray(Y[k], float)
#             y_main = y_main[np.isfinite(y_main)]
#             if y_main.size:
#                 m = np.nanmax(np.abs(y_main))
#                 if m == 0:
#                     lo, hi = -1.0, 1.0
#                 else:
#                     lo, hi = -1.1*m, 1.1*m

#                 if expand_only:
#                     # Only enlarge current limits
#                     cur_lo, cur_hi = ax.get_ylim()
#                     lo = min(lo, cur_lo)
#                     hi = max(hi, cur_hi)
#                 ax.set_ylim(lo, hi)

#     if created:
#         axes[-1].set_xlabel('Time')

#         # nice datetime ticks on shared x
#         locator   = mdates.AutoDateLocator()
#         formatter = mdates.ConciseDateFormatter(locator)
#         axes[-1].xaxis.set_major_locator(locator)
#         axes[-1].xaxis.set_major_formatter(formatter)

#         fig.suptitle('Relative position in RSW')

#     # Put a legend only on the top axes (one entry per overlay)
#     # (Call multiple times → legend will accumulate entries from overlays.)
#     axes[0].legend(loc='upper left', ncol=2)

#     return fig, axes

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm

def _overlay_colors(overlay_idx: int) -> list[str]:
    """Return 3 distinct colors for this overlay index."""
    # Use tab20 (20 distinct colors); take chunks of 3
    base = cm.get_cmap('tab20').colors  # length 20, tuples in 0..1
    L = len(base)
    start = (overlay_idx * 3) % L
    return [base[(start + i) % L] for i in range(3)]

def Residuals_RSW_Compare(residuals_RSW, time,
                       fig=None, axes=None,
                       label_prefix=None,
                       faint_alpha=0.50,
                       line_width_main=1.6,
                       line_width_other=1.0,
                       set_panel_limits=False,
                       expand_only=True,
                       overlay_idx=0):
    time_dt = ConvertToDateTime(time)

    created = False
    if fig is None or axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        created = True

    labels = [r'$\Delta R$', r'$\Delta S$', r'$\Delta W$']
    Y = [residuals_RSW[:, 0], residuals_RSW[:, 1], residuals_RSW[:, 2]]

    # ---- NEW: colors change per call
    colors = _overlay_colors(overlay_idx)

    # Distinguish overlays further with linestyles if you’d like
    linestyles = ['solid', '--', '-.', ':']
    ls = linestyles[overlay_idx % len(linestyles)]

    for k, ax in enumerate(axes):
        # plot all three; current component opaque, others faint
        for j in range(3):
            alpha = 1.0 if j == k else faint_alpha
            lw    = line_width_main if j == k else line_width_other
            z     = 3 if j == k else 1
            lab   = (f"{label_prefix} {labels[j]}" if label_prefix else f"{labels[j]}") if j == k else None
            if j == k:
                ax.plot(time_dt, Y[j],
                        color=colors[j], linestyle=ls,
                        alpha=alpha, lw=lw, zorder=z, label=lab)

        ax.set_ylabel(f'{labels[k]} [km]')
        ax.grid(True, alpha=0.3)

        if set_panel_limits:
            y_main = np.asarray(Y[k], float)
            y_main = y_main[np.isfinite(y_main)]
            if y_main.size:
                m = np.nanmax(np.abs(y_main))
                lo, hi = (-1.1*m, 1.1*m) if m > 0 else (-1.0, 1.0)
                if expand_only:
                    cur_lo, cur_hi = ax.get_ylim()
                    lo, hi = min(lo, cur_lo), max(hi, cur_hi)
                ax.set_ylim(lo, hi)

    if created:
        axes[-1].set_xlabel('Time')
        locator   = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(formatter)
        fig.suptitle('Relative position in RSW')

    # legend accumulates one entry per overlay (from focused row)
    axes[0].legend(loc='lower right')#, ncol=2)
    axes[1].legend(loc='lower right')
    axes[2].legend(loc='lower right')
    
    return fig, axes


# ##############################################################################################
# # RESIDUALS PER ITERATION
# ##############################################################################################
def Residuals_RMS(residuals_j2000):
    first_entry = 1 
    last_entry = 4
    if np.shape(residuals_j2000)[-1] == 3:
        print("processing absolute angular residuals...")
        first_entry = 1
        last_entry = 3
    iterations = np.shape(residuals_j2000)[0]
    rms = np.zeros(iterations)
    for i in range(iterations):
        A = residuals_j2000[i][:, first_entry:last_entry]          # (N,3) residuals (same RMS as inertial; RSW is a rotation)

        # unweighted scalar RMS (what Tudat prints)
        rms[i] = np.sqrt(np.mean(A**2))     

    fig_rms, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, len(rms)+1)  # if you haven't set x yet
    ax.scatter(x, rms)

    for i, v in enumerate(rms, start=1):
        ax.annotate(f"{v:.2e}", (i, v), textcoords="offset points", xytext=(0, 6), ha="center")

    ax.set_xticks(x)                               # only integer ticks (1..N)
    ax.set_xlim(0.5, len(rms) + 0.5)               # a bit of padding
    ax.set_xlabel('Iterations [-]')
    ax.set_ylabel('RMS Residual [m]')
    if last_entry == 3:
        ax.set_ylabel('RMS Residual [arcseconds]')
   
    ax.grid(True) 
    ax.set_title('RMS of Residuals per iteration')

    fig_rms.tight_layout()
    return fig_rms


# ##############################################################################################
# # FFT RESIDUALS FIG JONAS
# ##############################################################################################

def make_fft(residuals_data: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    signals = [residuals_data[:, 1], residuals_data[:, 2], residuals_data[:, 3]]
    sample_spacing = residuals_data[1:, 0] - residuals_data[:-1, 0]
    T = sample_spacing[0] if np.all(sample_spacing == sample_spacing[0]) else 0

    y_list = []
    for signal in signals:
        fft_signal = fft(signal)
        power_density = np.abs(fft_signal) ** 2
        y_list.append(power_density)

    N = len(residuals_data[:, 0])
    x = fftfreq(N, T)

    return y_list, len(y_list) * [x]


def create_fft_residual_figure(residuals_data,f_rot_hz):


    y_list, x_list = make_fft(residuals_data)
    assert len(y_list) == len(x_list)
    N = len(x_list[0])

    labels = ['r', 's', 'w']
    #colors, lead_color = get_band_color_scheme("ka_band")
    #colors = colors[0:2] + [colors[3]]


    fig, ax = plt.subplots(figsize=(10, 6))
    #pretty_grid(ax)

    for xf, yf, label in zip(x_list, y_list, labels):

        ax.plot(1/xf[1:N//2], yf[1:N//2], label=label, alpha=0.7)

    #ax.vlines((16*24*60*60), ymin=0, ymax=1, linestyles=':', color='grey', transform=ax.get_xaxis_transform())
    ax.vlines(f_rot_hz,ymin=0,ymax=1,linestyles=':',color='grey',transform=ax.get_xaxis_transform(), label = 'Triton Period')
    plt.legend(prop={'size': 12})

    ax.set_xlabel("freq [Hz]")
    ax.set_ylabel("signal strength [-]")

    ax.set_xscale('log')
    ax.set_yscale('log')

    year_in_s = 365*24*60*60
    month_in_s = 30*24*60*60
    day_in_s = 24*60*60
    hour_in_s = 60*60

    #ax.set_xlim([np.log10(day_in_s), np.log10(1000*year_in_s)])

    xtick_locs = ax.xaxis.get_majorticklocs()

    my_xtick_labels = []

    for x_loc in xtick_locs:

        x_val = x_loc

        if x_val > year_in_s:
            x_years = (x_val/year_in_s)
            my_xtick_labels.append(f"{x_years:.1f} years")

        elif x_val > month_in_s:
            x_month = (x_val / month_in_s)
            my_xtick_labels.append(f"{x_month:.1f} months")

        elif x_val > day_in_s:
            x_day = (x_val / day_in_s)
            my_xtick_labels.append(f"{x_day:.1f} days")

        else:
            x_hour = (x_val / hour_in_s)
            my_xtick_labels.append(f"{x_hour:.1f} hours")

    ax.xaxis.set_ticks(xtick_locs)
    ax.xaxis.set_ticklabels(my_xtick_labels)
    ax.tick_params(axis='x', labelrotation=25)

    #plt.show()

    return fig #,ax), #(1/xf[1:N//2], yf[1:N//2])



# ##############################################################################################
# # FFT RESIDUALS FIG Welch Segmenting ?
# ##############################################################################################

def psd_rsw(residuals_data, f_rot_hz, units='m',type='PSD'):
    """
    residuals_data: array with columns [t, R, S, W], t in seconds and (nearly) uniform.
    f_rot_hz: Triton spin/orbital frequency in Hz.
    """
    t = residuals_data[:, 0].astype(float)
    R = residuals_data[:, 1].astype(float)
    S = residuals_data[:, 2].astype(float)
    W = residuals_data[:, 3].astype(float)
    Y = [R, S, W]
    labels = [r'$\Delta R$', r'$\Delta S$', r'$\Delta W$']

    # Uniform sampling check
    dt = np.diff(t)
    if not np.allclose(dt, np.median(dt), rtol=1e-4, atol=0):
        raise ValueError("Time is not uniformly sampled; use a Lomb–Scargle periodogram instead.")
    fs = 1.0 / float(np.median(dt))  # Hz
    N = t.size

    # --- Choose robust segmenting params ---
    if N < 8:  # too short for Welch
        raise ValueError(f"Signal too short for Welch (N={N}).")

    nperseg = min(256, max(8, N // 2))      # strictly < N if N>=16
    if nperseg >= N:
        nperseg = max(8, N - 1)             # ensure at least 2 segments
    noverlap = nperseg // 2                  # 50% overlap
    win = get_window('hann', nperseg)

    fig, ax = plt.subplots(figsize=(10, 6))
    for y, lab in zip(Y, labels):
        y = y - np.mean(y)
        f, Pxx = welch(
            y, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap,
            detrend='constant', return_onesided=True, scaling='density'
        )  # PSD → (units)^2 / Hz

        # drop DC bin for log plots
        mask = f > 0

                
        if type == 'PSD':
            ax.plot(f[1:], Pxx[1:], label=lab, lw=1.2, alpha=0.9)  # drop DC bin
            ax.set_ylabel(f'PSD [{units}²/Hz]')
    
        elif type == 'ASD':
            ASD = np.sqrt(Pxx)
            ax.plot(f[1:], ASD[1:], label=lab, lw=1.2, alpha=0.9)
            ax.set_ylabel(f'ASD [{units}/√Hz]')

        #ax.plot(f[mask], Pxx[mask], label=lab, lw=1.2, alpha=0.9)

    # Triton mean-motion line (Hz)
    ax.vlines(f_rot_hz, ymin=0, ymax=1, linestyles=':', color='grey',
              transform=ax.get_xaxis_transform(), label='Triton mean motion')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('frequency [Hz]')
    #ax.set_ylabel(f'PSD [{units}²/Hz]')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig



# ##############################################################################################
# # FFT RESIDUALS FIG RAW
# ##############################################################################################

def periodogram_rsw(
    residuals_data,                 # array with columns [t, R, S, W]
    f_rot_hz,                       # Triton rotation/orbital frequency (Hz)
    units='m',                     # label for residual units (e.g., 'km' or 'm')
    mode='PSD',                     # 'PSD', 'ASD', or 'spectrum'
    window=None,                    # None -> boxcar; or e.g. 'hann', 'blackman'
    detrend='constant',             # 'constant', 'linear', or False
    zero_pad=1                      # 1=no padding; >1 to zero-pad N*zero_pad
):
    t = residuals_data[:, 0].astype(float)
    Y = [residuals_data[:, 1].astype(float),
         residuals_data[:, 2].astype(float),
         residuals_data[:, 3].astype(float)]
    labels = [r'$\Delta R$', r'$\Delta S$', r'$\Delta W$']

    # ---- sampling check
    dt = np.diff(t)
    if not np.allclose(dt, np.median(dt), rtol=1e-4, atol=0):
        raise ValueError("Non-uniform sampling: use a Lomb–Scargle periodogram.")
    fs = 1.0 / float(np.median(dt))   # Hz

    # window
    win = 'boxcar' if window is None else get_window(window, int(len(t)))

    # choose scaling
    if mode.upper() == 'PSD':
        scaling = 'density'           # -> units^2 / Hz
        y_label = f'PSD [{units}²/Hz]'
        post = lambda P: P
    elif mode.upper() == 'ASD':
        scaling = 'density'           # take sqrt of PSD
        y_label = f'ASD [{units}/√Hz]'
        post = np.sqrt
    elif mode.lower() == 'spectrum':
        scaling = 'spectrum'          # -> units^2
        y_label = f'Power spectrum [{units}²]'
        post = lambda P: P
    else:
        raise ValueError("mode must be 'PSD', 'ASD', or 'spectrum'")

    nfft = int(len(t) * max(1, int(zero_pad)))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for y, lab, c in zip(Y, labels, colors[:3]):
        # remove mean if detrending; keeps the DC bin clean
        f, P = periodogram(
            y, fs=fs, window=win, detrend=detrend,
            return_onesided=True, scaling=scaling, nfft=nfft
        )
        # drop DC for log plots
        mask = f > 0
        ax.plot(f[mask], post(P[mask]), label=lab, lw=1.2, alpha=0.9, color=c)
    
    df = 1.0 / (t[-1] - t[0])
    ax.axvspan(f_rot_hz - 0.5*df, f_rot_hz + 0.5*df,
           color='k', alpha=0.1, label='±½ bin at f_rot')


    # Triton mean-motion line (Hz)
    ax.vlines(f_rot_hz, ymin=0, ymax=1, linestyles=':', color='grey',
              transform=ax.get_xaxis_transform(), label='Triton mean motion')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel(y_label)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    fig.tight_layout()

    # --- usage (after your vline) ---
    add_special_tick_at(ax, f_rot_hz)





    return fig

def add_special_tick_at(ax, x0, decimals=2, emphasize=True, color="gray", space_before_e=True):
    """
    Add a real major tick at x0 on a log-x axis and label it with scientific notation
    rounded to `decimals` (default 2). Keeps existing log ticks.

    Call this AFTER ax.set_xscale('log') and after x-limits are final.
    """
    x0 = float(x0)
    if not np.isfinite(x0) or x0 <= 0:
        raise ValueError("x0 must be a positive finite number for a log-x axis.")

    # Ensure x0 is within the visible range (expand a bit if needed)
    xmin, xmax = ax.get_xlim()
    lo, hi = sorted([xmin, xmax])
    if not (lo < x0 < hi):
        lo = min(lo, x0 * 0.9)
        hi = max(hi, x0 * 1.1)
        ax.set_xlim(lo, hi)

    # Current major ticks (now in log scale); append x0 and lock them
    base_ticks = ax.get_xticks()
    ticks = np.sort(np.unique(np.append(base_ticks, x0)))
    ax.xaxis.set_major_locator(FixedLocator(ticks))

    # Build labels: normal log formatting except at x0 -> numeric with 2 decimals
    logfmt = LogFormatterSciNotation()
    special = f"{x0:.{decimals}e}"
    if space_before_e:
        special = special.replace("e", " e")

    labels = [special if np.isclose(t, x0, rtol=0, atol=abs(x0)*1e-12) else logfmt(t) for t in ticks]
    ax.xaxis.set_major_formatter(FixedFormatter(labels))

    # Emphasize the special tick label (optional)
    if emphasize:
        ax.figure.canvas.draw()  # ensure tick labels exist
        for txt in ax.get_xticklabels():
            if txt.get_text() == special:
                txt.set_fontweight("bold")
                txt.set_color(color)
                break

# ##############################################################################################
# # CARTESIAN DIFFERENCE
# ##############################################################################################

def PlotCartesianDifference(state_history_array1, state_history_array2, time_seconds=None):
    """
    Plot the difference in Cartesian position components between two state histories.
    """
    # --- Extract time
    time_dt = ConvertToDateTime(state_history_array1[:, 0] )

    # --- Extract positions (in km)
    r1_km = state_history_array1[:, 1:4] / 1e3
    r2_km = state_history_array2[:, 1:4] / 1e3

    # --- Compute differences
    diff = r1_km - r2_km  # Δx, Δy, Δz in km

    # --- Plot setup
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    labels = ['Δx [km]', 'Δy [km]', 'Δz [km]']

    for i in range(3):
        ax[i].plot(time_dt, diff[:, i])
        ax[i].set_ylabel(labels[i])
        ax[i].grid(True)

    #ax[-1].set_xlabel('Time [hours]')
    fig.suptitle('Cartesian Component Differences vs Time', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ax[-1].set_xlabel('Time')
    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax[-1].xaxis.set_major_locator(locator)
    ax[-1].xaxis.set_major_formatter(formatter)
    return fig
# ##############################################################################################
# # 3D PLOT
# ##############################################################################################
def Plot3D(pseudo_observations,state_history_array,states_SPICE):
    states_observations = pseudo_observations.get_observations()[0]
    states_observations_size = np.size(states_observations)/3
    states_observations = states_observations.reshape(int(states_observations_size),3)/1e3

    r_km = state_history_array[:,1:4]/1e3
    x1 = r_km[:,0]
    y1 = r_km[:,1]
    z1 = r_km[:,2]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Orbit path
    ax.plot(x1, y1, z1, lw=1.5,label='Triton Orbit Fitted')
    #ax.plot(x2, y2, z2, lw=1.5, label='Triton Orbit More Planets')
    ax.plot(states_SPICE[:,0]/1e3,states_SPICE[:,1]/1e3,states_SPICE[:,2]/1e3,lw=1.5,label='SPICE Triton Oribt',linestyle='dashed')

    ax.scatter(states_observations[:,0],states_observations[:,1],states_observations[:,2],lw=1.5,label='Observations')
    # Start and end points
    ax.scatter([x1[0]], [y1[0]], [z1[0]], s=40, label='Start', marker='o')
    ax.scatter([x1[-1]], [y1[-1]], [z1[-1]], s=40, label='End', marker='^')

    # Neptune at origin
    ax.scatter([0], [0], [0], s=80, label='Neptune', marker='*')

    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    ax.set_title('Triton orbit (Neptune-centered)')

    # Equal aspect ratio
    max_range = np.array([x1.max()-x1.min(), y1.max()-y1.min(), z1.max()-z1.min()]).max()
    mid_x = 0.5*(x1.max()+x1.min()); mid_y = 0.5*(y1.max()+y1.min()); mid_z = 0.5*(z1.max()+z1.min())
    ax.set_xlim(mid_x - 0.5*max_range, mid_x + 0.5*max_range)
    ax.set_ylim(mid_y - 0.5*max_range, mid_y + 0.5*max_range)
    ax.set_zlim(mid_z - 0.5*max_range, mid_z + 0.5*max_range)
    # try:
    #     ax.set_box_aspect([1,1,1])  # Matplotlib >=3.3
    # except Exception:
    #     pass

    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


def Plot3D_generic(state_history_array1,state_history_array2,states_SPICE):
    #states_observations = pseudo_observations.get_observations()[0]
    #states_observations_size = np.size(states_observations)/3
    #states_observations = states_observations.reshape(int(states_observations_size),3)/1e3

    r_km = state_history_array1[:,1:4]/1e3
    x1 = r_km[:,0]
    y1 = r_km[:,1]
    z1 = r_km[:,2]

    r_km = state_history_array2[:,1:4]/1e3
    x2 = r_km[:,0]
    y2 = r_km[:,1]
    z2 = r_km[:,2]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Orbit path
    ax.plot(x1, y1, z1, lw=1.5,label='Triton Orbit Simulated')
    ax.scatter(x2, y2, z2, lw=1.5, label='Triton Orbit Observations')
    ax.plot(states_SPICE[:,0]/1e3,states_SPICE[:,1]/1e3,states_SPICE[:,2]/1e3,lw=1.5,label='SPICE Triton Oribt',linestyle='dashed')

    #ax.scatter(states_observations[:,0],states_observations[:,1],states_observations[:,2],lw=1.5,label='Observations')
    # Start and end points
    ax.scatter([x1[0]], [y1[0]], [z1[0]], s=40, label='Start', marker='o')
    ax.scatter([x1[-1]], [y1[-1]], [z1[-1]], s=40, label='End', marker='^')

    # Neptune at origin
    ax.scatter([0], [0], [0], s=80, label='Neptune', marker='*')

    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    ax.set_title('Triton orbit (Neptune-centered)')

    # Equal aspect ratio
    max_range = np.array([x1.max()-x1.min(), y1.max()-y1.min(), z1.max()-z1.min()]).max()
    mid_x = 0.5*(x1.max()+x1.min()); mid_y = 0.5*(y1.max()+y1.min()); mid_z = 0.5*(z1.max()+z1.min())
    ax.set_xlim(mid_x - 0.5*max_range, mid_x + 0.5*max_range)
    ax.set_ylim(mid_y - 0.5*max_range, mid_y + 0.5*max_range)
    ax.set_zlim(mid_z - 0.5*max_range, mid_z + 0.5*max_range)
    # try:
    #     ax.set_box_aspect([1,1,1])  # Matplotlib >=3.3
    # except Exception:
    #     pass

    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


# ##############################################################################################
# # Function to compare RSW residuals from different simulations WIP
# ##############################################################################################
#Example Usage
#-----------------------------------------------------------------------------------------------------------------------------
# path_list = ["PoleOrientation/DefaultRotationModel/residuals_rsw.npy","PoleOrientation/SimpleRotationModel/residuals_rsw.npy"]
# label_list = ["Default Rotation Model","Simple Rotation Model"]
# fig,fig_diff = FigUtils.Compare_RSW_Different_Solutions(path_list,label_list)

# fig.savefig("R_RSW_Comparison_Simple_Rotation.pdf")
# fig_diff.savefig("Diff_R_RSW_Comparison_Simple_Rotation.pdf")

#-----------------------------------------------------------------------------------------------------------------------------


#Loads .npy file, extracts the last entry and converts to [km]
def LoadAndExtractResiduals(path):
    residuals = np.load(path)
    residuals_time = residuals[-1][:,0]
    residuals_final = residuals[-1][:,1:4]/1e3
    return residuals_time,residuals_final

def Compare_RSW_Different_Solutions(path_list,label_list):
    
    path = Path("/home/atn/Documents/Year 5/Thesis/Github/NeptuneTritonMasterThesis/Results")

    residuals_time = [None] * np.size(path_list)
    residuals_final = [None] * np.size(path_list)

    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for i in range(np.size(path_list)):
        residuals_time[i],residuals_final[i] = LoadAndExtractResiduals(path / path_list[i])
       
        fig,axes = Residuals_RSW_Compare(residuals_final[i], residuals_time[i],
                        fig=fig, axes=axes,
                        label_prefix=label_list[i],
                        overlay_idx=i)



    difference_residuals_rsw = residuals_final[0] - residuals_final[1]

    fig_diff = Residuals_RSW(difference_residuals_rsw,residuals_time[0],type="difference")

    return fig, fig_diff
    #fig.savefig("Residuals_RSW_Comparison_SSB.pdf")
    #fig_diff.savefig("Difference_Residuals_RSW_Neptune_SSB.pdf")



# ##############################################################################################
# # REAL OBSERVATIONS
# ##############################################################################################


def plot_RA_DEC_residuals(observation_times_DateFormat,
                          uncertainty_ra_arcseconds,
                          uncertainty_ra_initial_arcseconds,
                          uncertainty_dec_arcseconds,
                          uncertainty_dec_initial_arcseconds,
                          labels=["final","initial"],
                          save_path=None):
    """
    Plot initial and finall RA and DEC residuals side by side.

    Parameters
    ----------
    observation_times_DateFormat : array-like of datetime
    uncertainty_ra_arcseconds : array-like
    uncertainty_ra_initial_arcseconds : array-like
    uncertainty_dec_arcseconds : array-like
    uncertainty_dec_initial_arcseconds : array-like
    save_path : str or None, optional
        If provided, saves the figure to this path (e.g. 'SPICE_Residuals/RA_DEC_residuals.pdf')
    """




    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True, sharex=True)

    # --- RA plot ---
    ax[0].scatter(observation_times_DateFormat, uncertainty_ra_arcseconds, label=labels[0])
    ax[0].scatter(observation_times_DateFormat, uncertainty_ra_initial_arcseconds, label=labels[1])
    ax[0].set_xlabel('Observation epoch [years since ECLIPJ2000]')
    ax[0].set_ylabel('simulated - observed RA [arcseconds]')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc='lower right')

    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax[0].xaxis.set_major_locator(locator)
    ax[0].xaxis.set_major_formatter(formatter)

    # --- DEC plot ---
    ax[1].scatter(observation_times_DateFormat, uncertainty_dec_arcseconds, label=labels[0])
    ax[1].scatter(observation_times_DateFormat, uncertainty_dec_initial_arcseconds, label=labels[1])
    ax[1].set_xlabel('Observation epoch [years since ECLIPJ2000]')
    ax[1].set_ylabel('simulated - observed DEC [arcseconds]')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc='lower right')

    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig





# ##############################################################################################
# # Correlation plots
# ##############################################################################################

def plot_correlation_matrix(correlations, est_parameters, sh_degree=None, sh_order=None, 
                        title="Parameter Correlation Matrix", figsize=(12, 10)):
    """
        Plot correlation matrix with proper labels based on estimation parameters.
        
        Parameters:
        -----------
        correlations : array
            Correlation matrix with shape (n, n)
        est_parameters : list
            List of estimated parameters. Can include:
            - 'initial_state'
            - 'iau_rotation_model_pole'
            - 'iau_rotation_model_pole_rate'
            - 'gm_neptune'
            - 'gm_triton'
            - 'spherical_harmonics'
        sh_degree : int, optional
            Maximum degree for spherical harmonics (if included)
        sh_order : int, optional
            Maximum order for spherical harmonics (if included)
        title : str
            Plot title
        figsize : tuple
            Figure size (width, height)
        
        Returns:
        --------
        fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate parameter labels based on est_parameters
    labels = []
    
    for param in est_parameters:
        if param == 'initial_state':
            labels.extend(['x', 'y', 'z', 'vx', 'vy', 'vz'])
        
        elif param == 'iau_rotation_model_pole':
            labels.extend(['α₀', 'δ₀'])
        
        elif param == 'iau_rotation_model_pole_rate':
            labels.extend(['α̇₀', 'δ̇₀'])
        elif param == 'iau_rotation_model_pole_librations':
            labels.extend([[r'\alpha_{i}', r'\delta_{i}']])
            #units.extend(['rad', 'rad'])

        elif param == 'GM_Neptune':
            labels.append('GM_Nep')
        
        elif param == 'GM_Triton':
            labels.append('GM_Tri')
        
        elif param == 'spherical_harmonics':
            labels.extend(['C20', 'C40'])
            
    
    # Verify label count matches correlation matrix size
    n = correlations.shape[0]
    if len(labels) != n:
        print(f"Warning: Generated {len(labels)} labels but correlation matrix has size {n} in plot_correlation_matrix")
        # Pad with generic labels if needed
        if len(labels) < n:
            labels.extend([f'Param_{i}' for i in range(len(labels), n)])
        else:
            labels = labels[:n]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot correlation matrix
    im = ax.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    
    # Add correlation values as text
    for i in range(n):
        for j in range(n):
            corr_val = correlations[i, j]
            
            # Choose text color based on background
            text_color = 'white' if abs(corr_val) > 0.5 else 'black'
            
            # Format correlation value
            text = f'{corr_val:.2f}'
            
            ax.text(j, i, text, ha='center', va='center', 
                   color=text_color, fontsize=8, fontweight='bold')
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(n) - 0.5, minor=True)
    ax.set_yticks(np.arange(n) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def get_parameter_groups(est_parameters, correlations, sh_degree=None, sh_order=None):
    """
    Get parameter grouping information for better visualization.
    
    Parameters:
    -----------
    est_parameters : list
        List of estimated parameters
    correlations : array
        Correlation matrix
    sh_degree, sh_order : int, optional
        Spherical harmonics degree and order
    
    Returns:
    --------
    dict : Dictionary with parameter groups and their index ranges
    """
    groups = {}
    current_idx = 0
    
    for param in est_parameters:
        if param == 'initial_state':
            groups['Initial State'] = (current_idx, current_idx + 6)
            current_idx += 6
        
        elif param == 'iau_rotation_model_pole':
            groups['Pole Position'] = (current_idx, current_idx + 2)
            current_idx += 2
        
        elif param == 'iau_rotation_model_pole_rate':
            groups['Pole Rate'] = (current_idx, current_idx + 2)
            current_idx += 2
        elif param == 'iau_rotation_model_pole_librations':
            groups['Pole Librations'] = (current_idx,current_idx+2)
            #labels.extend([[r'\alpha_{i}', r'\delta_{i}']])
            current_idx += 2
        elif param == 'GM_Neptune':
            groups['GM Neptune'] = (current_idx, current_idx + 1)
            current_idx += 1
        
        elif param == 'GM_Triton':
            groups['GM Triton'] = (current_idx, current_idx + 1)
            current_idx += 1
        
        elif param == 'spherical_harmonics':
            groups['Spherical Harmonics'] = (current_idx, current_idx + 2)
            current_idx += 2
    
    return groups


# Example usage function
def plot_correlation_with_groups(correlations, est_parameters, sh_degree=None, sh_order=None):
    """
    Plot correlation matrix with group boundaries highlighted.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create the base correlation plot
    fig = plot_correlation_matrix(correlations, est_parameters, sh_degree, sh_order)
    ax = fig.axes[0]
    
    # Get parameter groups
    groups = get_parameter_groups(est_parameters, correlations, sh_degree, sh_order)
    
    # Draw group boundaries
    for group_name, (start, end) in groups.items():
        # Draw rectangles around parameter groups
        rect = plt.Rectangle((start - 0.5, start - 0.5), 
                            end - start, end - start,
                            fill=False, edgecolor='black', 
                            linewidth=2.5, linestyle='--')
        ax.add_patch(rect)
    
    return fig



# ##############################################################################################
# # Parameter plots
# ##############################################################################################


def plot_parameter_updates(parameter_updates, est_parameters, sh_degree=None, sh_order=None,
                          title="Parameter Updates from Initial to Best Iteration", figsize=(12, 8)):
    """
    Plot parameter updates with proper labels and units based on estimation parameters.
    
    Parameters:
    -----------
    parameter_updates : array
        Parameter differences (initial - best_iteration)
    est_parameters : list
        List of estimated parameters. Can include:
        - 'initial_state'
        - 'iau_rotation_model_pole'
        - 'iau_rotation_model_pole_rate'
        - 'gm_neptune'
        - 'gm_triton'
        - 'spherical_harmonics'
    sh_degree : int, optional
        Maximum degree for spherical harmonics (if included)
    sh_order : int, optional
        Maximum order for spherical harmonics (if included)
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    #if 'pole_ibrations_deg2' in est_parameters:
        
    # Generate parameter labels and units
    labels = []
    units = []
    groups = []  # For coloring by parameter type
    
    for param in est_parameters:
        if param == 'initial_state':
            labels.extend(['x', 'y', 'z', 'vx', 'vy', 'vz'])
            units.extend(['m', 'm', 'm', 'm/s', 'm/s', 'm/s'])
            groups.extend(['position'] * 3 + ['velocity'] * 3)
        
        elif param == 'iau_rotation_model_pole':
            labels.extend(['α₀', 'δ₀'])
            units.extend(['rad', 'rad'])
            groups.extend(['pole_position'] * 2)
        
        elif param == 'iau_rotation_model_pole_rate':
            labels.extend(['α̇₀', 'δ̇₀'])
            units.extend(['rad/s', 'rad/s'])
            groups.extend(['pole_rate'] * 2)
        
        elif param == 'iau_rotation_model_pole_librations':
            labels.extend([[r'\alpha_{i}', r'\delta_{i}']])
            units.extend(['rad', 'rad'])
            groups.extend(['pole_lib'] * 2)
        elif param == 'GM_Neptune':
            labels.append('GM_Nep')
            units.append('km³/s²')
            groups.append('gravity')
        
        elif param == 'GM_Triton':
            labels.append('GM_Tri')
            units.append('km³/s²')
            groups.append('gravity')
        
        elif param == 'spherical_harmonics':
            # Hardcoded for C20 and C40 only
            labels.extend(['C20', 'C40'])
            units.extend(['[-]', '[-]'])
            groups.extend(['spherical_harmonics', 'spherical_harmonics'])
            
    # Verify label count
    n = len(parameter_updates)
    if len(labels) != n:
        print(f"Warning: Generated {len(labels)} labels but have {n} parameters plot_parameter_updates")
        if len(labels) < n:
            labels.extend([f'Param_{i}' for i in range(len(labels), n)])
            units.extend(['[-]'] * (n - len(labels)))
            groups.extend(['other'] * (n - len(labels)))
        else:
            labels = labels[:n]
            units = units[:n]
            groups = groups[:n]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color mapping for different parameter groups
    color_map = {
        'position': '#1f77b4',      # Blue
        'velocity': '#ff7f0e',      # Orange
        'pole_position': '#2ca02c', # Green
        'pole_rate': '#d62728',     # Red
        'gravity': '#9467bd',       # Purple
        'spherical_harmonics': '#8c564b',  # Brown
        'other': '#7f7f7f'          # Gray
    }
    
    colors = [color_map.get(g, '#7f7f7f') for g in groups]
    
    # Create bar plot
    x = np.arange(n)
    bars = ax.bar(x, parameter_updates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Set labels
    ax.set_xlabel('Parameter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Update Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-ticks with parameter names and units
    ax.set_xticks(x)
    tick_labels = [f'{label}\n[{unit}]' for label, unit in zip(labels, units)]
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    
    # Add horizontal line at zero
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, parameter_updates)):
        height = bar.get_height()
        # Use scientific notation for very small values
        if abs(val) < 0.001 and val != 0:
            label_text = f'{val:.2e}'
        else:
            label_text = f'{val:.3f}'
        
        # Position text above or below bar depending on sign
        y_pos = height + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]) if height >= 0 else height - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        va = 'bottom' if height >= 0 else 'top'
        
        ax.text(i, y_pos, label_text, ha='center', va=va, fontsize=7, rotation=0)
    
    # Create legend for parameter groups
    from matplotlib.patches import Patch
    unique_groups = []
    legend_elements = []
    for group in groups:
        if group not in unique_groups:
            unique_groups.append(group)
            group_name = group.replace('_', ' ').title()
            legend_elements.append(Patch(facecolor=color_map.get(group, '#7f7f7f'), 
                                        edgecolor='black', label=group_name))
    
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def plot_parameter_history(parameter_history, est_parameters, best_iteration=None,
                           sh_degree=None, sh_order=None, 
                           title="Parameter Convergence History", figsize=(14, 10)):
    """
    Plot the evolution of all parameters through iterations.
    
    Parameters:
    -----------
    parameter_history : array
        Array with shape (n_params, n_iterations) containing parameter values at each iteration
    est_parameters : list
        List of estimated parameters
    best_iteration : int, optional
        Index of best iteration to highlight
    sh_degree, sh_order : int, optional
        Spherical harmonics degree and order
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate parameter labels
    labels = []
    groups = []
    
    for param in est_parameters:
        if param == 'initial_state':
            labels.extend(['x [km]', 'y [km]', 'z [km]', 'vx [km/s]', 'vy [km/s]', 'vz [km/s]'])
            groups.extend(['State'] * 6)
        elif param == 'iau_rotation_model_pole':
            labels.extend(['α₀ [rad]', 'δ₀ [rad]'])
            groups.extend(['Pole'] * 2)
        elif param == 'iau_rotation_model_pole_rate':
            labels.extend(['α̇₀ [rad/s]', 'δ̇₀ [rad/s]'])
            groups.extend(['Pole Rate'] * 2)
        elif param == 'iau_rotation_model_pole_librations':
            labels.extend([[r'\alpha_{i}', r'\delta_{i}']])
            #units.extend(['rad', 'rad'])
            groups.extend(['Pole Lib'] * 2)
        elif param == 'GM_Neptune':
            labels.append('GM_Nep [km³/s²]')
            groups.append('Gravity')
        elif param == 'GM_Triton':
            labels.append('GM_Tri [km³/s²]')
            groups.append('Gravity')
        elif param == 'spherical_harmonics':
            # Hardcoded for C20 and C40 only
            labels.extend(['C20', 'C40'])
            groups.extend(['spherical_harmonics', 'spherical_harmonics'])
          
    
    n_params = parameter_history.shape[0]
    n_iterations = parameter_history.shape[1]
    iterations = np.arange(n_iterations)
    
    # Create subplots - one per parameter
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for i in range(n_params):
        ax = axes[i]
        ax.plot(iterations, parameter_history[i, :], marker='o', linewidth=1.5, markersize=4)
        
        # Highlight best iteration
        if best_iteration is not None:
            ax.axvline(best_iteration, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Best Iteration')
            ax.plot(best_iteration, parameter_history[i, best_iteration], 'r*', markersize=15, label='Best Value')
        
        ax.set_ylabel(labels[i] if i < len(labels) else f'Param {i}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{groups[i] if i < len(groups) else "Parameter"}', fontsize=10, fontweight='bold')
        
        if best_iteration is not None and i == 0:
            ax.legend(fontsize=8)
    
    # Remove empty subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])
    
    # Set x-labels for bottom row
    for ax in axes[n_params - n_cols:n_params]:
        ax.set_xlabel('Iteration', fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ##############################################################################################
# # TWO RMS Plots
# ##############################################################################################


def plot_rms_comparison(rms_SPICE, rms_Norm=None, title="RMS Comparison Across Estimation Scenarios", 
                        figsize=(14, 6)):
    """
    Plot RMS values for SPICE and Norm as line plots with markers.
    
    Parameters:
    -----------
    rms_SPICE : dict
        Dictionary with estimation scenarios as keys and RMS values
    rms_Norm : dict
        Dictionary with estimation scenarios as keys and RMS values
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib figure
    """
    # import matplotlib.pyplot as plt
    # import numpy as np
    
    # Get keys (should be the same for both dicts)
    keys = list(rms_SPICE.keys())
    
    # Extract values
    spice_values = [rms_SPICE[k] for k in keys]
    if rms_Norm != None:
        norm_values = [rms_Norm[k] for k in keys]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(keys))
    
    # Plot lines with markers
    ax.plot(x, spice_values, marker='o', linewidth=2, markersize=8, 
            label='RMS SPICE', color='steelblue', alpha=0.8)
    if rms_Norm != None:
        ax.plot(x, norm_values, marker='s', linewidth=2, markersize=8, 
                label='RMS Norm', color='coral', alpha=0.8)
        
    # Labels and formatting
    ax.set_ylabel('RMS', fontsize=12, fontweight='bold')
    ax.set_xlabel('Estimation Scenario', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # # Add value labels at each point
    if rms_Norm != None:
        for i, (s_val, n_val) in enumerate(zip(spice_values, norm_values)):
            # SPICE label
            label_s = f'{s_val:.2e}' if s_val < 0.01 else f'{s_val:.3f}'
            ax.text(i, s_val, label_s, ha='center', va='bottom', 
                fontsize=7, fontweight='bold', color='steelblue')
            
            # Norm label
            label_n = f'{n_val:.2e}' if n_val < 0.01 else f'{n_val:.3f}'
            ax.text(i, n_val, label_n, ha='center', va='top', 
                fontsize=7, fontweight='bold', color='coral')
        
    else:
        for i, s_val in enumerate(spice_values):
            # SPICE label
            label_s = f'{s_val:.2e}' if s_val < 0.01 else f'{s_val:.3f}'
            ax.text(i, s_val, label_s, ha='center', va='bottom', 
                fontsize=7, fontweight='bold', color='steelblue')
            
        

    plt.tight_layout()
    
    return fig

        # print("Next")

# #---------------------------------------------------------------------------------------
# PARAMETER PLOTS (Y axis) VS ESTIMATION NAME (X axis)
# #---------------------------------------------------------------------------------------


def plot_state_magnitude_updates(best_parameter_update, 
                                 title="Position and Velocity Magnitude Updates",
                                 figsize=(14, 6)):
    """
    Plot magnitude of position and velocity updates (first 6 parameters).
    
    Parameters:
    -----------
    best_parameter_update : dict
        Dictionary with keys as scenario names and values as parameter update arrays.
        First 6 elements are assumed to be [x, y, z, vx, vy, vz] in meters and m/s.
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    keys = list(best_parameter_update.keys())
    
    # Extract position and velocity updates and convert to km and km/s
    delta_r_magnitudes = []
    delta_v_magnitudes = []
    
    for key in keys:
        updates = best_parameter_update[key]
        
        # Position: x, y, z (convert from m to km)
        x, y, z = updates[0] / 1000, updates[1] / 1000, updates[2] / 1000
        delta_r = np.sqrt(x**2 + y**2 + z**2)
        delta_r_magnitudes.append(delta_r)
        
        # Velocity: vx, vy, vz (convert from m/s to km/s)
        vx, vy, vz = updates[3] / 1000, updates[4] / 1000, updates[5] / 1000
        delta_v = np.sqrt(vx**2 + vy**2 + vz**2)
        delta_v_magnitudes.append(delta_v)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(keys))
    
    # Plot 1: Delta R magnitude
    axes[0].plot(x, delta_r_magnitudes, marker='o', linewidth=2, markersize=8, 
                color='steelblue', alpha=0.8)
    axes[0].set_ylabel('|Δr| [km]', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Estimation Scenario', fontsize=12, fontweight='bold')
    axes[0].set_title('Position Magnitude Update', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, v in enumerate(delta_r_magnitudes):
        label = f'{v:.3f}'
        axes[0].text(i, v, label, ha='center', va='bottom', 
                    fontsize=8, fontweight='bold')
    
    # Plot 2: Delta V magnitude
    axes[1].plot(x, delta_v_magnitudes, marker='s', linewidth=2, markersize=8, 
                color='coral', alpha=0.8)
    axes[1].set_ylabel('|Δv| [km/s]', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Estimation Scenario', fontsize=12, fontweight='bold')
    axes[1].set_title('Velocity Magnitude Update', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, v in enumerate(delta_v_magnitudes):
        label = f'{v:.6f}'
        axes[1].text(i, v, label, ha='center', va='bottom', 
                    fontsize=8, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_state_component_updates(best_parameter_update,
                                  title="Individual State Component Updates",
                                  figsize=(14, 12)):
    """
    Plot individual components of position and velocity updates (6 subplots).
    
    Parameters:
    -----------
    best_parameter_update : dict
        Dictionary with keys as scenario names and values as parameter update arrays.
        First 6 elements are assumed to be [x, y, z, vx, vy, vz] in meters and m/s.
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    keys = list(best_parameter_update.keys())
    
    # Extract all components and convert to km and km/s
    x_vals = [best_parameter_update[k][0] / 1000 for k in keys]  # m -> km
    y_vals = [best_parameter_update[k][1] / 1000 for k in keys]  # m -> km
    z_vals = [best_parameter_update[k][2] / 1000 for k in keys]  # m -> km
    vx_vals = [best_parameter_update[k][3] / 1000 for k in keys]  # m/s -> km/s
    vy_vals = [best_parameter_update[k][4] / 1000 for k in keys]  # m/s -> km/s
    vz_vals = [best_parameter_update[k][5] / 1000 for k in keys]  # m/s -> km/s
    
    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    x = np.arange(len(keys))
    
    # Component data and labels
    # Now organized as: Left column = Position (X, Y, Z), Right column = Velocity (Vx, Vy, Vz)
    # Each row: (position_component, velocity_component)
    components = [
        # Row 0: X position (left), X velocity (right)
        [(x_vals, 'Δx [km]', 'X Position Update', 'steelblue', 'o'),
         (vx_vals, 'Δvx [km/s]', 'X Velocity Update', 'coral', 'o')],
        # Row 1: Y position (left), Y velocity (right)
        [(y_vals, 'Δy [km]', 'Y Position Update', 'steelblue', 's'),
         (vy_vals, 'Δvy [km/s]', 'Y Velocity Update', 'coral', 's')],
        # Row 2: Z position (left), Z velocity (right)
        [(z_vals, 'Δz [km]', 'Z Position Update', 'steelblue', '^'),
         (vz_vals, 'Δvz [km/s]', 'Z Velocity Update', 'coral', '^')],
    ]
    
    # Plot each component
    for row_idx, row_components in enumerate(components):
        for col_idx, (vals, ylabel, subplot_title, color, marker) in enumerate(row_components):
            ax = axes[row_idx, col_idx]
            
            ax.plot(x, vals, marker=marker, linewidth=2, markersize=8,
                    color=color, alpha=0.8)
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(subplot_title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add value labels
            for i, v in enumerate(vals):
                # Use appropriate precision based on column (left=position, right=velocity)
                if col_idx == 0:  # Position components (km)
                    label = f'{v:.3f}'
                else:  # Velocity components (km/s)
                    label = f'{v:.6f}'
                va = 'bottom' if v >= 0 else 'top'
                ax.text(i, v, label, ha='center', va=va,
                        fontsize=7, fontweight='bold')
    
    # Set x-labels only on bottom row
    for ax in axes[2, :]:
        ax.set_xlabel('Estimation Scenario', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha='right', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_state_updates_combined(best_parameter_update):
    """
    Create both plots: magnitude updates and component updates.
    
    Parameters:
    -----------
    best_parameter_update : dict
        Dictionary with keys as scenario names and values as parameter update arrays
    
    Returns:
    --------
    tuple : (fig_magnitude, fig_components)
    """
    fig1 = plot_state_magnitude_updates(best_parameter_update)
    fig2 = plot_state_component_updates(best_parameter_update)
    
    return fig1, fig2

#A parameter (y axis) vs simulation name (x axis)
def plot_parameter_update(best_parameter_update,simulations,parameter_to_plot,libration_deg='deg1', 
                                 title="Position and Velocity Magnitude Updates",
                                 figsize=(14, 6)):
    """
    Plot any parameter.
    
    Parameters:
    -----------
    best_parameter_update : dict
        Dictionary with keys as scenario names and values as parameter update arrays.
        First 6 elements are assumed to be [x, y, z, vx, vy, vz] in meters and m/s.
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig : matplotlib figure
    """

    keys = list(best_parameter_update.keys())
    
    param_names = ['initial_state','GM_Neptune','GM_Triton',
                   'iau_rotation_model_pole','iau_rotation_model_pole_rate','iau_rotation_model_pole_librations'
                   'spherical_harmonics']
    param_lengths = [6, 1, 1, 2, 2, 2, 2]
    
    plot_param_1 = []
    plot_name_1 = []
    plot_param_2 = []
    plot_name_2 = []
    for key in keys:
        #pole_pos_and_libration_amplitude_Pole_Jacobson2009
        updates = best_parameter_update[key]
        est_parameters = simulations[key]['est_parameters']
        parameters_indicies = est_parameters_indicies[key]
        if parameter_to_plot in est_parameters:        
                if parameter_to_plot == 'iau_rotation_model_pole':
                    index_in_updates = parameters_indicies.index('pole_position')
                if parameter_to_plot == 'iau_rotation_model_pole_rate':
                    index_in_updates = parameters_indicies.index('pole_rate')
                if parameter_to_plot == 'iau_rotation_model_pole_librations':
                    if libration_deg == 'deg1':
                        index_in_updates = parameters_indicies.index('pole_lib')
                    elif libration_deg == 'deg2' and 'pole_librations_deg2' in est_parameters:
                        index_in_updates = parameters_indicies.index('pole_lib_deg2')
                    else:
                        continue #does not append indicies for edge case where iau_rotation_model_pole_librations exists but pole_librations_deg2 does not
                plot_param_1.append(updates[index_in_updates])
                plot_name_1.append(key)
                plot_param_2.append(updates[index_in_updates+1])
                plot_name_2.append(key)

        # #Edge case GM Plots
        # if parameter_to_plot == 'GM':
        #     if 'GM_Neptune' in est_parameters:
        #         plot_param_1.append(updates[6])
        #         plot_name_1.append(key)
        #     if 'GM_Triton' in est_parameters:
        #         idx = est_parameters.index('GM_Triton')
        #         plot_param_2.append(updates[5+idx])
        #         plot_name_2.append(key)
        

    if parameter_to_plot == 'iau_rotation_model_pole':
        labels = [r'$\alpha_0$ [rad]', r'$\delta_0$ [rad]']
    elif parameter_to_plot == 'iau_rotation_model_pole_rate':
        labels = [r'$\dot{\alpha}_0$ [rad/s]', r'$\dot{\delta}_0$ [rad/s]']
    elif parameter_to_plot == 'iau_rotation_model_pole_librations':
        if libration_deg == 'deg1':
            labels = [r'$\alpha_1$ [rad]', r'$\delta_1$ [rad]']
        elif libration_deg == 'deg2':
            labels = [r'$\alpha_2$ [rad]', r'$\delta_2$ [rad]']
    elif parameter_to_plot == 'GM':
        labels = [r'$GM_{\text{Nep}}$ [m$^3$/s$^2$]',r'$GM_{\text{Tri}}$ [m$^3$/s$^2$]']
    elif parameter_to_plot == 'GM_Triton':
        labels = [r'$GM_{\text{Tri}}$ [m$^3$/s$^2$]']
    elif parameter_to_plot == 'spherical_harmonics':
        labels = ['$C_{20}$', '$C_{40}$']



    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(keys))
    
    # Plot 1: Delta Param 1
    axes[0].plot(plot_name_1, plot_param_1, marker='o', linewidth=2, markersize=8, 
                color='steelblue', alpha=0.8)
    axes[0].set_ylabel(labels[0], fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Estimation Scenario', fontsize=12, fontweight='bold')
    #axes[0].set_title('Position Magnitude Update', fontsize=13, fontweight='bold')
    axes[0].set_xticks(plot_name_1)
    axes[0].set_xticklabels(plot_name_1, rotation=45, ha='right', fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    # for i, v in enumerate(plot_param_1):
    #     label = f'{v:.3f}'
    #     axes[0].text(i, v, label, ha='center', va='bottom', 
    #                 fontsize=8, fontweight='bold')
    
    # Plot 2: Delta V magnitude
    axes[1].plot(plot_name_2, plot_param_2, marker='s', linewidth=2, markersize=8, 
                color='coral', alpha=0.8)
    axes[1].set_ylabel(labels[1], fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Estimation Scenario', fontsize=12, fontweight='bold')
    #axes[1].set_title('Velocity Magnitude Update', fontsize=13, fontweight='bold')
    axes[1].set_xticks(plot_name_2)
    axes[1].set_xticklabels(plot_name_2, rotation=45, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    # for i, v in enumerate(plot_param_2):
    #     label = f'{v:.6f}'
    #     axes[1].text(i, v, label, ha='center', va='bottom', 
    #                 fontsize=8, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


##########################################################################################
# PLOT declination and right ascension plot 
##########################################################################################
def plot_pole_movement(time_column,alpha_array,delta_array,title='Pole Movement vs Time'):
    """
    Plot pole movement model (IAU 2015) for right ascension and declination.
    
    Parameters:
    -----------
    time_column : np.ndarray
        Time in seconds from J2000 epoch, shape (N, 1) or (N,)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object containing the plots
    """
    # J2000 epoch: January 1, 2000, 12:00:00 TT
    J2000_datetime = datetime(2000, 1, 1, 12, 0, 0)

    # Convert to degrees
    alpha_deg = np.rad2deg(alpha_array)
    delta_deg = np.rad2deg(delta_array)
    
    # Convert time_column to datetime objects
    time_seconds = time_column.flatten()
    time_dates = [J2000_datetime + timedelta(seconds=float(t)) for t in time_seconds]
    
    # # Downsample if data is too dense (optional)
    # if len(time_seconds) > 10000:
    #     step = len(time_seconds) // 10000
    #     time_dates = time_dates[::step]
    #     alpha_deg = alpha_deg[::step]
    #     delta_deg = delta_deg[::step]
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Right Ascension
    ax1.plot(time_dates, alpha_deg, linewidth=1.5)
    ax1.set_ylabel('Right Ascension α (degrees)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    #ax1.set_title('Pole Movement Model (IAU 2015)', fontsize=14, fontweight='bold')
    
    # Plot Declination
    ax2.plot(time_dates, delta_deg, linewidth=1.5)
    ax2.set_ylabel('Declination δ (degrees)', fontsize=12)
    ax2.set_xlabel('Time (Year-Month)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis to show year-month
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    return fig
