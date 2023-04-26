'''In the following code, Experiment 1 = 220322, Experiment 2 = 160822 spheroid 1, Experiment 3 = 160822 spheroid 2,
Experiment 4 = 240322, Experiment 5 = 120822, Experiment 6 = 091122 and Experiment 7 = 171122.'''

import numpy as np
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

directory = './data/invading/'
filenames = os.listdir(directory)
sns.set_theme(context='talk', font_scale=0.7)

def input_csv(filename): # read raw position data files
    df = pd.read_csv(directory + filename)
    df = df[['TrackID', 'Time', 'Position X', 'Position Y', 'Position Z']] # TrackID is a unique cell identifier

    # rescale as required
    if filename in ['120822.csv', '160822 spheroid 1.csv', '160822 spheroid 2.csv', '091122.csv', '171122.csv']:
        unit = 0.325
    else:
        unit = 1
    df['Position X'] = df['Position X'] * unit
    df['Position Y'] = df['Position Y'] * unit

    # set invasion time for invading spheroids
    if filename == '220322.csv':
        onset = 12
    elif filename == '240322.csv':
        onset = 75
    elif filename == '160822 spheroid 1.csv':
        onset = 23
    elif filename == '160822 spheroid 2.csv':
        onset = 35
    else:
        onset = 0 # set as 0 for non-invading spheroids
    df = df[df['Time'] > onset]
    return df

def drop(file, threshold): # data cleaning
    file.drop(file[file['TrackID'].isnull()].index, inplace=True)
    cells = file['TrackID'].unique()
    for cell in cells:
        cell_file = file[file['TrackID'] == cell]
        if len(cell_file) < threshold: # drop rows with false signals
            file.drop(file[file['TrackID'] == cell].index, inplace=True)

    x_c, y_c = file['Position X'].mean(), file['Position Y'].mean()
    time = file['Time'].unique()
    for t in time:
        time_file = file[file['Time'] == t]
        time_file['2D dis to C'] = np.sqrt((time_file['Position X'] - x_c) ** 2 + (time_file['Position Y'] - y_c) ** 2)
        mean_dis = time_file['2D dis to C'].mean()
        std_dis = time_file['2D dis to C'].std()
        far_cells = time_file[time_file['2D dis to C'] > mean_dis + 3 * std_dis] # exclude cells that are too far away from centre of spheroid
        ids = far_cells['TrackID'].unique()
        for id in ids:
            file.drop(file[(file['Time'] == t) & (file['TrackID'] == id)].index, inplace=True)
    return file

'''Temporal autocorrelation analysis'''
def velocity_temp(file, dt):
    cells = file['TrackID'].unique()
    file_velocity = []
    for cell in cells:
        cell_file = file[file['TrackID'] == cell].sort_values(by='Time')
        cell_file.drop_duplicates(subset="Time", inplace=True)
        # velocity at time t
        cell_file['dt'] = cell_file['Time'].diff()
        cell_file['vx_t'] = cell_file['Position X'].diff() / cell_file['dt']
        cell_file['vy_t'] = cell_file['Position Y'].diff() / cell_file['dt']
        cell_file['vz_t'] = cell_file['Position Z'].diff() / cell_file['dt']
        # velocity at time t + dt
        timepoints = cell_file['Time']
        vx_dt = []
        vy_dt = []
        vz_dt = []
        for t in timepoints:
            time_slice = cell_file[cell_file['Time'] == t + dt]
            if time_slice.empty: # for timepoints with no corresponding dt position data
                vx_dt.append(None)
                vy_dt.append(None)
                vz_dt.append(None)
                continue
            ind = time_slice.index.to_numpy()[0]
            vx_dt.append(time_slice.at[ind, 'vx_t'])
            vy_dt.append(time_slice.at[ind, 'vy_t'])
            vz_dt.append(time_slice.at[ind, 'vz_t'])
        cell_file['vx_dt'] = vx_dt
        cell_file['vy_dt'] = vy_dt
        cell_file['vz_dt'] = vz_dt
        file_velocity.append(cell_file.dropna())
    file_v = pd.concat(file_velocity, ignore_index=True)

    time = sorted(file_v['Time'].unique())
    file_norm = []
    for t in time:
        time_file = file_v[file_v['Time'] == t]
        # for velocity at time t
        # calculate centroid velocity
        vcx, vcy, vcz = time_file['vx_t'].mean(), time_file['vy_t'].mean(), time_file['vz_t'].mean()
        # calculate relative velocity
        time_file['vx_t'] = time_file['vx_t'] - vcx
        time_file['vy_t'] = time_file['vy_t'] - vcy
        time_file['vz_t'] = time_file['vz_t'] - vcz
        # normalise velocity
        mag = np.sqrt(time_file['vx_t'] ** 2 + time_file['vy_t'] ** 2 + time_file['vz_t'] ** 2)
        time_file['vx_t'] = time_file['vx_t'] / mag
        time_file['vy_t'] = time_file['vy_t'] / mag
        time_file['vz_t'] = time_file['vz_t'] / mag
        time_file['v_t'] = time_file.apply(lambda x: np.array([x['vx_t'], x['vy_t'], x['vz_t']]), axis=1)
        # for velocity at time t + dt
        # calculate centroid velocity
        vcx, vcy, vcz = time_file['vx_dt'].mean(), time_file['vy_dt'].mean(), time_file['vz_dt'].mean()
        # calculate relative velocity
        time_file['vx_dt'] = time_file['vx_dt'] - vcx
        time_file['vy_dt'] = time_file['vy_dt'] - vcy
        time_file['vz_dt'] = time_file['vz_dt'] - vcz
        # normalise velocity
        mag = np.sqrt(time_file['vx_dt'] ** 2 + time_file['vy_dt'] ** 2 + time_file['vz_dt'] ** 2)
        time_file['vx_dt'] = time_file['vx_dt'] / mag
        time_file['vy_dt'] = time_file['vy_dt'] / mag
        time_file['vz_dt'] = time_file['vz_dt'] / mag
        time_file['v_dt'] = time_file.apply(lambda x: np.array([x['vx_dt'], x['vy_dt'], x['vz_dt']]), axis=1)
        file_norm.append(time_file.dropna())
    file_v = pd.concat(file_norm, ignore_index=True)
    return file_v

def correlation_temp(file):
    file['inner'] = file.apply(lambda x: np.inner(x['v_t'], x['v_dt']), axis=1) # inner product between adjacent timepoints in each cell
    corr = file['inner'].mean()
    return corr

def plot_temp(file, time_range, name): # input desired range of dt
    data = pd.DataFrame({'Time': list(time_range)})
    corr = []
    for t in time_range:
        inner = correlation_temp(velocity_temp(file, t)) # plot correlation against lag time
        corr.append(inner)
    data['Correlation'] = corr

    # curve fitting
    model = lambda x,a,b,c: a * np.exp(-x/b) + c # fit exponential decay curve
    par_opt, par_cov = curve_fit(model, data['Time'], data['Correlation'])
    data['Fitted'] = model(data['Time'], *par_opt)
    
    # plot observed and fitted curves
    plt.figure()
    plt.ylim(-0.02, 0.07)
    s = sns.scatterplot(data=data, x='Time', y='Correlation', marker='x')
    sns.lineplot(data=data, x='Time', y='Fitted', color='red', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(par_opt)) # do not fit to non-invading
    s.set_xticks(range(0, 71, 10))
    s.set_xticklabels(list(range(0, 71, 10)))
    s.set_yticks([-0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    s.set_yticklabels([-0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    s.set(xlabel='dt (3 min)', ylabel='Correlation', title='Experiment 1')
    plt.legend(loc='upper right', frameon=False)
    plt.show()

def plot_log_temp(files, names): # files is a list of dataframes, names is a list of filenames
    data = pd.DataFrame({'Time': list(range(2, 71))})
    colours = ['red', 'orange', 'green', 'blue']
    labels = ['Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 4']
    plt.figure()
    plt.ylim(-4.0, -1.0)
    for ind, file in enumerate(files):
        filename = names[ind]
        corr = []
        mid = file['Position Z'].median()
        # file = file[file['Position Z'] <= mid] # select top
        # file = file[file['Position Z'] > mid] # select bottom
        for t in range(2, 71):
            inner = correlation_temp(velocity_temp(file, t)) # correlation values for all files
            corr.append(inner)
        # smoothing
        data[filename + '_smoothed'] = savgol_filter(corr, 31, 1)
        # smoothed curve in log scale
        pos_data = data[data[filename + '_smoothed'] > 0]
        pos_data[filename + '_log'] = np.log10(pos_data[filename + '_smoothed'])
        l = sns.lineplot(data=pos_data, x='Time', y=filename + '_log', label=labels[ind], color=colours[ind])
    l.set_xticks(range(0, 71, 10))
    l.set_xticklabels(list(range(0, 71, 10)))
    l.set_yticks([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0])
    l.set_yticklabels([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0])
    l.set(xlabel='dt (3 min)', ylabel='Log of correlation', title='Overall')
    plt.legend(loc='upper right', frameon=False) # overall and top plots
    # plt.legend(loc='lower left', frameon=False) # bottom plot
    plt.show()

files, names = [], []
for file in filenames:
    if file.endswith('.csv'):
        files.append(drop(input_csv(file), 3))
        names.append(file[:-4])
plot_log_temp(files, names)

def get_char_timescale(file, time_range, name):
    mid = file['Position Z'].median()
    top = file[file['Position Z'] <= mid]
    bottom = file[file['Position Z'] > mid]
    data, data_top, data_bottom = pd.DataFrame({'Time': list(time_range)}), pd.DataFrame({'Time': list(time_range)}), pd.DataFrame({'Time': list(time_range)})
    corr, corr_top, corr_bottom = [], [], []
    for t in time_range:
        inner = correlation_temp(velocity_temp(file, t))
        corr.append(inner)
        inner_top = correlation_temp(velocity_temp(top, t))
        corr_top.append(inner_top)
        inner_bottom = correlation_temp(velocity_temp(bottom, t))
        corr_bottom.append(inner_bottom)
    data['Correlation'] = corr
    data_top['Correlation'] = corr_top
    data_bottom['Correlation'] = corr_bottom    

    # smoothing
    data['Smoothed'] = savgol_filter(data['Correlation'], 31, 1)
    data_top['Smoothed'] = savgol_filter(data_top['Correlation'], 31, 1)
    data_bottom['Smoothed'] = savgol_filter(data_bottom['Correlation'], 31, 1)

    # smoothed curve in log scale
    pos_data = data[data['Smoothed'] > 0]
    pos_data['Logarithm'] = np.log10(pos_data['Smoothed'])
    pos_data_top = data_top[data_top['Smoothed'] > 0]
    pos_data_top['Logarithm'] = np.log10(pos_data_top['Smoothed'])
    pos_data_bottom = data_bottom[data_bottom['Smoothed'] > 0]
    pos_data_bottom['Logarithm'] = np.log10(pos_data_bottom['Smoothed'])

    # coefficients of linear fit
    # overall
    sub_data = pos_data[pos_data['Time'] <= 20] # fit linear curve up to dt = 20
    x = sub_data['Time']
    y = sub_data['Logarithm']
    gradient, intercept = np.polyfit(x, y, 1) # linear fit
    print(f"Coefficients for top of sample {name}:", gradient, intercept)
    # top
    sub_data_top = pos_data_top[pos_data_top['Time'] <= 20]
    x_top = sub_data_top['Time']
    y_top = sub_data_top['Logarithm']
    gradient, intercept = np.polyfit(x_top, y_top, 1)
    print(f"Coefficients for top of sample {name}:", gradient, intercept)
    # bottom
    sub_data_bottom = pos_data_bottom[pos_data_bottom['Time']<=20]
    x_bottom = sub_data_bottom['Time']
    y_bottom = sub_data_bottom['Logarithm']
    gradient, intercept = np.polyfit(x_bottom, y_bottom, 1)
    print(f"Coefficients for bottom of sample {name}:", gradient, intercept)

'''Spatial correlation analysis'''
def velocity_spat(file):
    # get relative, normalised 2D velocity
    cells = file['TrackID'].unique()
    file_velocity = []
    for cell in cells:
        cell_file = file[file['TrackID'] == cell].sort_values(by='Time')
        cell_file.drop_duplicates(subset="Time", inplace=True)
        cell_file['dt'] = cell_file['Time'].diff()
        cell_file['vx'] = cell_file['Position X'].diff() / cell_file['dt']
        cell_file['vy'] = cell_file['Position Y'].diff() / cell_file['dt']
        file_velocity.append(cell_file.dropna())
    file_v = pd.concat(file_velocity, ignore_index=True)

    time = sorted(file_v['Time'].unique())
    file_norm = []
    for t in time:
        time_file = file_v[file_v['Time'] == t]
        # calculate centroid velocity
        vcx, vcy = time_file['vx'].mean(), time_file['vy'].mean()
        # calculate relative velocity
        time_file['vx'] = time_file['vx'] - vcx
        time_file['vy'] = time_file['vy'] - vcy
        # normalise velocity
        mag = np.sqrt(time_file['vx'] ** 2 + time_file['vy'] ** 2)
        time_file['vx'] = time_file['vx'] / mag
        time_file['vy'] = time_file['vy'] / mag
        # calculate radial (unit) vector
        center_x, center_y = time_file['Position X'].mean(), time_file['Position Y'].mean()
        time_file['vrx'] = time_file['Position X'] - center_x
        time_file['vry'] = time_file['Position Y'] - center_y
        mag = np.sqrt(time_file['vrx'] ** 2 + time_file['vry'] ** 2)
        time_file['vrx'] = time_file['vrx'] / mag
        time_file['vry'] = time_file['vry'] / mag
        # velocity - radial velocity
        time_file['vx'] = time_file['vx'] - time_file['vrx']
        time_file['vy'] = time_file['vy'] - time_file['vry']
        # calculate rotation velocity
        vrotx, vroty = time_file['vx'].mean(), time_file['vy'].mean()
        time_file['vx'] = time_file['vx'] - vrotx
        time_file['vy'] = time_file['vy'] - vroty
        # normalise velocity
        mag = np.sqrt(time_file['vx'] ** 2 + time_file['vy'] ** 2)
        time_file['vx'] = time_file['vx'] / mag
        time_file['vy'] = time_file['vy'] / mag
        time_file['Velocity'] = time_file.apply(lambda x: np.array([x['vx'], x['vy']]), axis=1)
        file_norm.append(time_file.dropna())
    file_v = pd.concat(file_norm, ignore_index=True)
    return file_v

def correlation_spat(file, t): # input desired timepoint
    file_t = file[file['Time'] == t]
    pair_file = pd.DataFrame()

    # correlation between all neighbour pairs
    dr = []
    inner = []
    for ind_i, row_i in file_t.iterrows():
        for ind_j, row_j in file_t.iterrows():
            if ind_i < ind_j:
                dr.append(math.sqrt((row_i['Position X'] - row_j['Position X']) ** 2 + (row_i['Position Y'] - row_j['Position Y']) ** 2))
                inner.append(np.inner(row_i['Velocity'], row_j['Velocity']))
    pair_file['dr'] = dr
    pair_file['inner'] = inner

    # group inner product values into bins based on distance between neighbours
    pair_file['bin'] = pd.cut(pair_file['dr'], bins=np.linspace(pair_file['dr'].min(), pair_file['dr'].max(), 100), include_lowest=True)

    # take average correlation within each bin
    bins = sorted(pair_file['bin'].unique())
    corr_file = pd.DataFrame()
    dist = []
    corr = []
    for bin in bins:
        bin_file = pair_file[pair_file['bin']==bin]
        dist.append(bin_file['dr'].mean())
        corr.append(bin_file['inner'].mean())
    corr_file['Distance'] = dist
    corr_file['Correlation'] = corr
    return corr_file

def plot_spat(file, name):
    mid = file['Position Z'].median()
    # file = file[file['Position Z'] <= mid] # select top
    # file = file[file['Position Z'] > mid] # select bottom
    data = correlation_spat(velocity_spat(file), 40) # plot correlation against dr
    plt.figure()
    plt.ylim(-1.00, 1.00)
    s = sns.scatterplot(data=data, x='Distance', y='Correlation', marker='.')
    s.set_xticks(range(0, 121, 20))
    s.set_xticklabels(list(range(0, 121, 20)))
    s.set_yticks([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
    s.set_yticklabels([-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
    s.set(xlabel='Distance (\u03bcm)', ylabel='Correlation', title='Experiment 6')
    plt.show()