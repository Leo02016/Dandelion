
from pandas import read_csv
from pandas import read_excel
import numpy as np
import pandas as pd
import os, re
import matplotlib.pyplot as plt


def smape(ts_test, ts_pred):
    err = 2 * np.abs(ts_pred - ts_test) / (np.abs(ts_pred) + np.abs(ts_test))
    err[np.logical_and(ts_pred == 0, ts_test == 0)] = 0
    err = np.mean(err)
    return err


def mape(ts_test, ts_pred):
    err = np.abs(ts_pred - ts_test) / np.abs(ts_test)
    err[np.logical_and(ts_pred == 0, ts_test == 0)] = 0
    err[err == float('Inf')] = 10000
    return np.mean(err)


def mase(ts_test, ts_pred, freq, ts_train):
    train_naive_mae = np.mean(np.abs(ts_train[freq:] - ts_train[:-freq]))
    test_mae = np.mean(np.abs(ts_pred - ts_test))

    return test_mae / train_naive_mae


def med_abs(ts_test, ts_pred):
    err = np.median(np.abs(ts_test-ts_pred))
    return np.mean(err)



def draw_figure():

    path_1 = './MNA/'
    path_2 = './DAMAN/'
    path_3 = './DAMAN_M/'
    path_4 = './DAMAN_D/'
    path_5 = './bi-lstm/'
    path_6 = './ARIMAX/'
    path_7 = './MVR/'
    ID = [3994, 42706, 29753, 4927, 29450, 43579]
    num = 8
    input_file = 'test.csv'
    header = ['y_hat', 'Y', 'ID', 'Date']
    smape = [[[] for _ in range(num)] for _ in range(len(ID))]
    abs = [[[] for _ in range(num)] for _ in range(len(ID))]
    Dates = [[[] for _ in range(num)] for _ in range(len(ID))]
    for i in range(len(ID)):
        df_MNA = read_csv(path_1 + input_file)[header]
        df_DAMAN = read_csv(path_2 + input_file)[header]
        df_DAMAN_M = read_csv(path_3 + input_file)[header]
        df_DAMAN_D = read_csv(path_4 + input_file)[header]
        df_bi_lstm = read_csv(path_5 + input_file)[header]
        df_ARIMAX = read_csv(path_6 + input_file)[header]
        df_MVR = read_csv(path_7 + input_file)[header]
        df_MNA = df_MNA[df_MNA['ID'] == ID[i]]
        df_DAMAN = df_DAMAN[df_DAMAN['ID'] == ID[i]]
        df_DAMAN_M = df_DAMAN_M[df_DAMAN_M['ID'] == ID[i]]
        df_DAMAN_D = df_DAMAN_D[df_DAMAN_D['ID'] == ID[i]]
        df_bi_lstm = df_bi_lstm[df_bi_lstm['ID'] == ID[i]]
        df_ARIMAX = df_ARIMAX[df_ARIMAX['ID'] == ID[i]]
        df_MVR = df_MVR[df_MVR['ID'] == ID[i]]
        df_500 = df_DAMAN
        quarter = ['Q1', 'Q2', 'Q3', 'Q4']
        for j, df in enumerate([df_DAMAN, df_DAMAN_M, df_DAMAN_D, df_ARIMAX, df_bi_lstm, df_MNA, df_MVR, df_500]):
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            for year in range(2015, 2019):
                for a, month in enumerate([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]):
                    if year == 2015 and a <=2:
                        continue
                    if j == 7:
                        v1, v2 = val_metrics_baseline(df, month, year)
                    else:
                        v1, v2 = val_metrics(df, month, year)
                    smape[i][j].append(v1)
                    abs[i][j].append(v2)
                    Dates[i][j].append('{} {}'.format(year-2000, quarter[a]))
    for i in range(len(ID)):
        print(ID[i])
        # df_MNA, df_DAMAN, df_DAMAN_M, df_DAMAN_D, df_bi_lstm, df_ARIMAX, df_MVR
        plt.plot(Dates[i][0], np.log(abs[i][0]), marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.plot(Dates[i][0], np.log(abs[i][1]), marker='v', linestyle='-', linewidth=2, markersize=6)
        plt.plot(Dates[i][0], np.log(abs[i][2]), marker='>', linestyle='-', linewidth=2, markersize=6)
        plt.plot(Dates[i][0], np.log(abs[i][3]), marker='1', linestyle='-', linewidth=2, markersize=6)
        plt.plot(Dates[i][0], np.log(abs[i][4]), marker='s', linestyle='--', linewidth=2, markersize=6)
        plt.plot(Dates[i][0], np.log(abs[i][5]), marker='D', linestyle='-.', linewidth=2, markersize=6)
        plt.plot(Dates[i][0], np.log(abs[i][6]), marker='x', linestyle=':', linewidth=2, markersize=6)
        plt.plot(Dates[i][0], np.log(abs[i][7]), marker='h', linestyle='-', linewidth=2, markersize=6)
        plt.plot(Dates[i][0], np.log(smape[i][j]))
        plt.xticks(rotation=30, fontsize=12)
        plt.yticks(fontsize=16)
        plt.ylabel('Logarithmic Med_abs', fontsize=16)
        plt.ylim([-8, 4])
        # Please following the legend following order
        plt.legend(['Dandelion', 'Dandelion-M', 'Dandelion-D', 'ARIMAX', 'Bi-LSTM', 'MNA', 'MVR', 'ConEst'], loc='upper left', ncol=3, fontsize=12)
        # plt.legend(['Dandelion', 'ARIMAX', 'ConEst'], loc='upper left', ncol=3, fontsize=12)
        # plt.show()
        plt.savefig('{}_abs.jpg'.format(ID[i]), figsize=(1000, 1000))
        plt.close()
    # for i in range(len(ID)):
    #     print(ID[i])
    #     plt.plot(Dates[i][0], np.log(smape[i][0]), marker='o', linestyle='-', linewidth=2, markersize=6)
    #     # plt.plot(Dates[i][0], np.log(smape[i][1]), marker='v', linestyle='-', linewidth=2, markersize=6)
    #     # plt.plot(Dates[i][0], np.log(smape[i][2]), marker='>', linestyle='-', linewidth=2, markersize=6)
    #     plt.plot(Dates[i][0], np.log(smape[i][3]), marker='1', linestyle='-', linewidth=2, markersize=6)
    #     # plt.plot(Dates[i][0], np.log(smape[i][4]), marker='s', linestyle='--', linewidth=2, markersize=6)
    #     # plt.plot(Dates[i][0], np.log(smape[i][5]), marker='D', linestyle='-.', linewidth=2, markersize=6)
    #     # plt.plot(Dates[i][0], np.log(smape[i][6]), marker='x', linestyle=':', linewidth=2, markersize=6)
    #     plt.plot(Dates[i][0], np.log(smape[i][7]), marker='h', linestyle='-', linewidth=2, markersize=6)
    #     plt.xticks(rotation=30, fontsize=12)
    #     plt.yticks(fontsize=16)
    #     plt.ylabel('Logarithmic Smape', fontsize=16)
    #     plt.ylim([-3, 2])
    #     # plt.legend(['Dandelion', 'Dandelion-M', 'Dandelion-D', 'ARIMAX', 'Bi-LSTM', 'MNA', 'MVR', 'ConEst'], loc='upper left', ncol=3, fontsize=12)
    #     plt.legend(['Dandelion', 'ARIMAX', 'ConEst'], loc='upper left', ncol=3, fontsize=12)
    #     # plt.show()
    #     plt.savefig('{}_smape.jpg'.format(ID[i]), figsize=(1000, 1000))
        plt.close()


def val_metrics(df, month, year):
    dates = [list(map(int, re.split('-|/', '{}'.format(date[:10])))) for date in
             df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))]
    indices = np.array([i if date[0] == year and date[1] in month else 0 for i, date in enumerate(dates)])
    indices = indices.nonzero()[0]
    temp_df = df.iloc[indices]
    if len(indices) == 0:
        return 0.0, 0.0
    else:
        return smape(temp_df['Y'], temp_df['y_hat']), med_abs(temp_df['Y'], temp_df['y_hat']) * 50


def val_metrics_baseline(df, month, year):
    dates = [list(map(int, re.split('-|/', '{}'.format(date[:10])))) for date in
             df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))]
    indices = np.array([i if date[0] == year and date[1] in month else 0 for i, date in enumerate(dates)])
    indices = indices.nonzero()[0]
    temp_df = df.iloc[indices]
    zeros = np.zeros(temp_df['y_hat'].shape)
    if len(indices) == 0:
        return 0.0, 0.0
    else:
        return smape(temp_df['Y'], zeros), med_abs(temp_df['Y'], zeros) * 50


if __name__ == '__main__':
    draw_figure()
