import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
from scipy import signal
from scipy.optimize import leastsq
import pylab as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
from statistics import StatisticsError


mpl.rc("font", size=7)

PATH = "C:/Users/Mark/Desktop/Toriyama/USGS/Brandywine/"
PATH2 = "C:/Users/Mark/Desktop/Toriyama/USGS/01115170/"

def format_data():

    df = pd.read_csv(PATH + "GHO.txt", comment="#", parse_dates=True, skiprows=[28], header=0,
                     index_col=0, delimiter="\t", usecols=[2, 4], names=["Datetime", "Gage_Height"])


    dup = df[df.index.duplicated()]
    df.drop(index=dup.index, inplace=True)
    df = df.resample("10T").asfreq()
    df.interpolate(method="time", limit=1, inplace=True)
    lows = df["Gage_Height"].resample("Y").apply(lambda x: min(x))
    # print(lows)
    low_avg = lows[pd.to_datetime("2014-12-31"): pd.to_datetime("2019-12-31")].agg("mean")
    df.loc[pd.to_datetime("2013-09-24 14:20:00")] = low_avg
    df.interpolate(method="slinear", inplace=True)
    # df = df.rolling(4320, win_type="boxcar").mean()
    missing = df[df.isna().any(axis=1)].copy()
    missing.fillna("E", inplace=True)
    df.dropna(inplace=True)
    missing.to_csv(PATH + "01115170_GHO_missing.csv")
    df.to_csv(PATH + "01115170_GHO_format.csv")


def plot_file(file, s=False, col=None):
    if s:
        df = pd.read_csv(PATH + file)
    else:
        if col is None:
            df = pd.read_csv(PATH + file, index_col=0, parse_dates=True)
        else:
            df = pd.read_csv(PATH + file, index_col=0, parse_dates=True, usecols=[0, col])
    pyplot.plot(df)
    pyplot.show()


def fit_line():

    df = pd.read_csv(PATH + "01115170_GHO_format.csv", index_col=0, parse_dates=True)
    # df = df.resample("M", closed="left", label="left").asfreq()
    N = len(df.index)  # number of data points
    t = np.linspace(0, 4 * np.pi, N)
    f = 1.15247  # Optional!! Advised not to use

    guess_mean = np.mean(df["Gage_Height"].values)
    print(guess_mean)
    guess_std = 3 * np.std(df["Gage_Height"].values) / (2 ** 0.5) / (2 ** 0.5)
    print(guess_std)
    guess_phase = np.pi/2
    print(guess_phase)
    guess_freq = 3
    guess_amp = 0.47
    data_first_guess = guess_std * np.sin(t + guess_phase) + guess_mean
    optimize_func = lambda x: x[0] * np.sin(x[1] * t + x[2]) + x[3] - df["Gage_Height"].values
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func,
                                                     np.array([guess_amp, guess_freq, guess_phase, guess_mean]))[0]

    data_fit = est_amp * np.sin(est_freq * t + est_phase) + est_mean
    fine_t = np.arange(0, max(t), 0.1)
    data_fit = est_amp * np.sin(est_freq * fine_t + est_phase) + est_mean

    plt.plot(t, df, '.')
    plt.plot(t, data_first_guess, label='first guess')
    plt.plot(fine_t, data_fit, label='after fitting')
    plt.legend()
    plt.show()


def fourier():

    df = pd.read_csv(PATH + "01115170_GHO_format.csv", index_col=0, parse_dates=True)
    samp = len(df.index)
    wind = signal.windows.boxcar(samp)
    mod = df["Gage_Height"][:samp] * wind
    transform = np.fft.fft(mod)
    pyplot.plot(2/samp * np.abs(transform[:21]))
    pyplot.show()




def filter_data(band, type, fr_plot=False, apply=True, p=True, cor_test=False, c_plot=False):

    fig = None
    domain = None
    intervals = None
    xscale = 500
    n = 40001
    loss = int(n/2)

    b = signal.firwin(n, band, window="blackmanharris", pass_zero=type, fs=2000)

    if fr_plot:
        fig = pyplot.figure()
        w, h = signal.freqz(b, fs=2000)
        domain = w/xscale
        yvals = 20 * np.log10(abs(h))
        ax = fig.add_subplot(title="%s %s" % (str(band), type))
        ax.plot(domain, yvals)
        for f in band:
            ax.axvline(f/xscale, min(yvals), color="red", linewidth=0.5)
        fig.tight_layout()
        pyplot.show()
    if apply:
        df = pd.read_csv(PATH + "Brandywine_GHO.csv", index_col=0, parse_dates=True)
        # df = df.iloc[173000: 440000]
        filtered = pd.DataFrame(index=df.index[:-loss], columns=df.columns, dtype="float64")
        for item in df.columns:
            sig = df[item].values
            prep = np.zeros(loss)
            sig = np.append(prep, sig)
            y = signal.convolve(sig, b, mode='valid')
            filtered[item] = y
        fig = pyplot.figure()
        ax1 = fig.add_subplot(211, title="Original")
        ax1.plot(df["Gage_Height"])
        ax2 = fig.add_subplot(212, title="Filtered", sharex=ax1)
        ax2.plot(filtered["Gage_Height"])
        fig.tight_layout()
        pyplot.show()
        if p:
            for item in filtered.columns:
                if item == "Precip_Total" or item == "Snow_Depth":
                    filtered[item] = filtered[item].mask(filtered[item] < 0, other=0)
            filtered = filtered.applymap(lambda x: round(x, 3))
            filtered.to_csv(PATH + "Brandywine_" + type.split("pass")[0] + ".csv")
        if cor_test:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scale_vals = scaler.fit_transform(filtered.values)
            for n1 in range(len(filtered.columns)):
                filtered.iloc[:, n1] = scale_vals[:, n1]
            w_level = filtered[filtered.columns[0]].values
            stats = {}
            auto = []
            ref_pk = None
            if c_plot:
                fig = pyplot.figure()
                domain = np.arange(-len(w_level) / 2, len(w_level) / 2)
                intervals = np.arange(-3000, 3001, 500)
            for n2 in range(len(filtered.columns)):
                to_compare = filtered[filtered.columns[n2]].values
                cor = signal.correlate(w_level, to_compare, mode="same")
                if n2 == 0:
                    auto = cor
                else:
                    mae = mean_absolute_error(auto, cor)
                peaks, pk_data = signal.find_peaks(cor, height=(None, None))
                for n3 in range(len(peaks)):
                    try:
                        if pk_data["peak_heights"][n3] == pk_data["peak_heights"].max():
                            if n2 == 0:
                                ref_pk = [peaks[n3], pk_data["peak_heights"][n3], mean(np.diff(peaks))]
                            else:

                                mx_pk = [peaks[n3], pk_data["peak_heights"][n3], mean(np.diff(peaks))]
                                pk_shift = abs(ref_pk[0] - mx_pk[0])/96
                                frq_diff = abs(ref_pk[2] - mx_pk[2])/96
                                pk_offs = min(abs(peaks - len(w_level) / 2))/96
                                pk_diff = ref_pk[1] - mx_pk[1]
                                stats[filtered.columns[n2]] = [round(pk_shift, 4), round(frq_diff, 4), round(pk_offs, 4),
                                                               round(pk_diff, 4), round(mae, 4)]
                    except StatisticsError:
                        stats = "Error"
                        break
                    except TypeError:
                        stats = "Error"
                        break

                if c_plot:
                    axn = fig.add_subplot(len(filtered.columns), 1, n2 + 1, title=filtered.columns[n2])
                    axn.plot(domain, cor)
                    axn.vlines(intervals, 0, max(cor) + 0.1*max(cor), linestyles="dashed", colors=["red"])
            if c_plot:
                fig.tight_layout()
                pyplot.show()
            return stats



if __name__ == "__main__":

    # plot_file("Brandywine_ref.csv")
    df = pd.read_csv(PATH + "Brandywine_GHO.csv", index_col=0, parse_dates=True)
    df = df.resample("W", closed="left", label="left").min()

    # plot_file("Brandywine_GHO.csv", False, 1)
    # x1 = 1/8770
    # x2 = 1/8750
    # print(filter_data([x1, x2], "bandpass", p=True, cor_test=False, c_plot=False, fr_plot=True))

    # df = pd.read_csv(PATH + "Brandywine_test.csv", index_col=0, parse_dates=True)
    # idx = df.index
    # df = df["Gage_Height"]
    # intervals = [x for x in range(0, len(df), 30)]
    # dt = signal.detrend(df, bp=intervals)
    # dt = pd.DataFrame(dt, index=idx)
    # dt = signal.savgol_filter(dt, 25, 3)
    # df1 = pd.read_csv(PATH + "schuylkill_high.csv")
    # df1 = df1["Gage_Height"].values
    # df2 = pd.read_csv(PATH + "schuylkill_low.csv")
    # df2 = df2["Gage_Height"].values
    # # print(filter_data([3.0], "lowpass", p=True, cor_test=False, c_plot=False, fr_plot=True))
    # b = signal.firwin(1201, [3.0], window="blackmanharris", pass_zero="highpass", fs=2000)
    # test= signal.convolve(df1, df2)
    # # dt = signal.detrend(df["Gage_Height"].values, bp=[0, 500, 1000, 1500, 2000])
    # # rec, rem = signal.deconvolve(df["Gage_Height"].values, dt)
    # # print(rec)
    # # print(rem)
    smooth = signal.savgol_filter(df["Gage_Height"].values, 55, 3)
    # smooth = pd.DataFrame(smooth, index=df.index)
    # df["Gage_Height"] = smooth
    # smooth = df.resample("D").max()
    # df.dropna(inplace=True)
    # smooth = signal.savgol_filter(df["Gage_Height"].values, 3, 1)
    # smooth = pd.DataFrame(smooth, index=df.index)
    # dt = signal.detrend(smooth)
    # smooth2 = signal.savgol_filter(smooth, 7, 1)
    # smooth = np.gradient(smooth)
    fig = pyplot.figure()
    ax1 = fig.add_subplot(211, title="Original")
    ax1.plot(df["Gage_Height"].values)
    ax2 = fig.add_subplot(212, title="Detrend", sharex=ax1)
    ax2.plot(smooth)
    fig.tight_layout()
    pyplot.show()
    # df = pd.read_csv(PATH + "schuylkill_phil_diff_rs_shift.csv", index_col=0, parse_dates=True, usecols=[0, 1, 8])
    # df = pd.read_csv(PATH2 + "01115170_clean.csv", index_col=0, parse_dates=True, usecols=[0, 1, 3])
    # values = df.values
    # # ensure all data is float
    # # values = values.astype('float32')
    # # normalize features
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled = scaler.fit_transform(values)
    # df1 = df["Gage_Height"].values
    # df2 = df["Air_Temp"].values
    # cor = signal.correlate(values[:, 0], values[:, 1], mode="same")
    # peaks, pk_data = signal.find_peaks(cor, height=(0.1, None))
    # peaks = (peaks - 1376)
    # # print(peaks)
    # # print(pk_data)
    # print(pk_data["peak_heights"].max())
    # print(list(zip(peaks, pk_data["peak_heights"])))
    # center = int(len(cor)/2)
    # xax = [x for x in range(-center, center )]
    # intervals = np.arange(-2000, 2000, 400)
    # pyplot.plot(xax, cor)
    # pyplot.vlines(intervals, min(cor) - 0.1 * min(cor), max(cor) + 0.1 * max(cor), linestyles="dashed", colors=["red"])
    # pyplot.show()

    # print(filter_data([2.0], "highpass", p=True, cor_test=False, c_plot=False, fr_plot=True))
