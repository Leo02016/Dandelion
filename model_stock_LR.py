import numpy as np
import pandas as pd
import os, argparse, re
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression



def linear_regression(X,y):
    weight = np.ones((X.shape[1],1))
    alpha = 0.25
    beta = 0.6
    iter = 0
    while lambda_sqr(X, y, weight)/2 >= 0.0001:
        iter += 1
        stepsize = NewtonMethod(X, y, weight, alpha, beta, 1)
        weight = weight + stepsize * delta_x(X, y, weight)
    mean_square_error = pow(np.linalg.norm(y - np.dot(X, weight)), 2)/len(y)
    return mean_square_error, weight


def NewtonMethod(X, y, weight, alpha, beta, stepsize):
    newton_step = delta_x(X, y, weight)
    while function_value(X, y, weight + stepsize * newton_step) > function_value(X, y, weight)\
            + alpha * stepsize * np.dot(gradient(X, y, weight).T, newton_step)[0]:
        stepsize = stepsize * beta
    return stepsize

def lambda_sqr(X, y, weight):
    return -np.dot(gradient(X, y, weight).T, delta_x(X, y, weight))


def function_value(X, y, weight):
    return pow(np.linalg.norm(y - np.dot(X, weight)), 2)


def gradient(X, y, weight):
    return 2 * np.dot(np.dot(X.transpose(), X), weight) - 2 * np.dot(X.transpose(), y).transpose().T


def hessian(X):
    return 2 * np.dot(X.transpose(), X)


def delta_x(X, y, weight):
    A = hessian(X)
    b = gradient(X, y, weight)
    return -np.linalg.solve(A, b)

def main():
    parser = argparse.ArgumentParser(description='Use NMF model to extract useful features')
    parser.add_argument('-g', dest='gpu', type=int, default=0,
                        help='The index of GPU to use if there are multiple GPUs. The default is 0.')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    path = './../Attention_NN/data/'
    output_path = './../Attention_NN/res/multiview_regression/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    # id_list = []
    # with open('./../Attention_NN/res/id_list.txt', 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         id_list.append(list(map(int, line.split())))
    # id_list = np.concatenate(id_list)
    id_list = [8960]
    suffix = '_all'
    for file in os.listdir(path):
        df = pd.read_csv(path + file)
        new_headers = list(df.columns.values)
        for id in df['ID'].unique():
            writer = pd.ExcelWriter(output_path + '{}_regression_prediction.xlsx'.format(file[:-4]))
            if id not in id_list:
                continue
            if os.path.isfile(output_path + '{}_regression_prediction.xlsx'.format(file[:-4])):
                df_1 = pd.read_excel(output_path + '{}_regression_prediction.xlsx'.format(file[:-4]))
                if id in df_1['ID'].unique():
                    print('Already training the model for the stock id = {}'.format(id))
                    continue
            print('Start training the model for the stock id = {}'.format(id))
            headers = list(df.columns.values)
            indices = np.where(df['ID'].values == id)
            temp_df = df.loc[indices]
            train_index = []
            test_index = []
            temp_df['Date'] = pd.to_datetime(temp_df['Date']).dt.date
            dates = temp_df['Date'].values
            for i in range(len(dates)):
                date = list(map(int, re.split('-|/', '{}'.format(dates[i]))))
                idx = np.argmax(date)
                # check which idx is year
                if idx == 0:
                    if date[0] >= 2015:
                        if date[1] >= 9 or date[0] >= 2016:
                            test_index.append(i)
                        else:
                            train_index.append(i)
                    else:
                        train_index.append(i)
                else:
                    if date[2] >= 2015:
                        if date[0] >= 9 or date[2] >= 2016:
                            test_index.append(i)
                        else:
                            train_index.append(i)
                    else:
                        train_index.append(i)
            X = temp_df[new_headers].values
            X = normalize(X, axis=0, norm='max')
            Y = temp_df['Y'].values
            Y = Y.reshape((Y.shape[0], 1))
            Y = Y.astype(float)
            # split data into training data and test data based on the sampling date.
            x_train = X[train_index]
            x_test = X[test_index]
            y_train = Y[train_index]
            y_test = Y[test_index]


            reg = LinearRegression().fit(x_train, y_train)
            weight = reg.coef_.T
            mean_square_error, weight = linear_regression(x_train, y_train)
            new_y = np.array([np.dot(a, weight) for a in x_train], dtype=np.float32)
            bias = np.mean(y_train) - np.mean(new_y)
            new_y = new_y + bias
            n = 100
            # auto-regression transformation
            new_X_2 = np.zeros((new_y.shape[0], n))
            new_X_2[0, n - 1] = new_y[0, 0]
            for j in range(0, new_y.shape[0]):
                if n > j:
                    new_X_2[j, n - j: n] = Y[0: j, 0]
                    new_X_2[j, n - 1] = new_y[j, 0]
                else:
                    new_X_2[j, :] = Y[j - n:j, 0]
                    new_X_2[j, n - 1] = new_y[j, 0]
            reg2 = LinearRegression().fit(new_X_2, y_train)

            # for prediction
            # weight = reg.coef_.T
            new_y = np.array([np.dot(a, weight) for a in x_test], dtype=np.float32)
            bias = np.mean(y_test) - np.mean(new_y)
            new_y = new_y + bias

            # auto-regression transformation
            new_X_2 = np.zeros((new_y.shape[0], n))
            new_X_2[0, n - 1] = new_y[0, 0]
            for j in range(0, new_y.shape[0]):
                if n > j:
                    new_X_2[j, n - j: n] = Y[0: j, 0]
                    new_X_2[j, n - 1] = new_y[j, 0]
                else:
                    new_X_2[j, :] = Y[j - n:j, 0]
                    new_X_2[j, n - 1] = new_y[j, 0]

            # auto-regression
            weight2 = reg2.coef_
            new_y = np.array([np.dot(weight2, a) for a in new_X_2], dtype=np.float32)
            bias2 = np.mean(y_test) - np.mean(new_y)
            y_hat = new_y + bias2
            results = np.concatenate([temp_df[headers].values[test_index], np.array(y_hat)], axis=1)
            headers = headers + ['y_hat']
            if os.path.isfile(output_path + '{}_regression_prediction.xlsx'.format(file[:-4])):
                df_1 = pd.read_excel(output_path + '{}_regression_prediction.xlsx'.format(file[:-4]))
                df_2 = pd.DataFrame(data=results, columns=headers)
                frames = [df_1, df_2]
                final_frames = pd.concat(frames)
            else:
                final_frames = pd.DataFrame(data=results, columns=headers)
            final_frames.to_excel(writer, file[:-4])
            writer.save()


if __name__ == "__main__":
    main()
