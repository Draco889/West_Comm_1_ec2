from math import sqrt
from statistics import mean
from numpy import concatenate
from numpy import zeros
from numpy import empty
from numpy import reshape
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import PReLU
from keras.layers import Dropout
from keras.losses import Huber
from keras.layers import Flatten
from keras.layers import LSTM
from keras import activations
from keras.initializers import Constant
from scipy.signal import detrend


class Model:

    def __init__(self, r_PATH, d_PATH, s_length, n_train, n_batch, epochs, step, cols, med=None, diff=True, topf=False,
                 p=False, deriv=False):

        self.s_length = s_length
        self.n_train_hours = n_train
        self.n_batch = n_batch
        self.epochs = epochs
        self.step = step
        self.topf = topf
        self.diff = diff
        self.med = med
        self.deriv = deriv

        self.dataset = read_csv(d_PATH, header=0, index_col=0, usecols=cols)
        self.ref_data = read_csv(r_PATH, header=0, index_col=0, usecols=[0, 1])
        if med is not None:
            for col_no in range(len(self.dataset.columns)):
                if self.dataset.columns[col_no] == "Med_Diff":
                    self.med_col = col_no
        self.feat_no = self.dataset.shape[1]
        values = self.dataset.values
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(values)
        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, self.s_length, 1)

        # drop columns we don't want to predict
        to_drop = []
        to_drop += [x for x in range((self.dataset.shape[1] * self.s_length) + 1,
                                     (self.dataset.shape[1]) * (self.s_length + 1))]  # note parenthesis placement
        if topf:
            to_drop += [x for x in range(0, self.s_length * self.feat_no, self.feat_no)]
            self.feat_no -= 1
        reframed.drop(reframed.columns[to_drop], axis=1, inplace=True)

        if p:
            print(reframed.head())
        self.values = reframed.values

        self.train_X, self.train_y, self.test_X, self.test_y = self.split_data()
        self.model, self.history = self.fit_model()

    @staticmethod
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def split_data(self):

        train = self.values[:self.n_train_hours, :]
        test = self.values[self.n_train_hours:, :]
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, s_length, features]
        train_X = train_X.reshape((train_X.shape[0], self.s_length, int(train_X.shape[1] / self.s_length)))
        test_X = test_X.reshape((test_X.shape[0], self.s_length, int(test_X.shape[1] / self.s_length)))

        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        return train_X, train_y, test_X, test_y

    def fit_model(self):
        drp = 0.25
        n = 200
        # design network
        model = Sequential()
        # model.add(Flatten())
        # model.add(Dense(70, input_shape=(self.train_X.shape[1], self.train_X.shape[2]), activation="relu"))
        model.add(LSTM(n, input_shape=(self.train_X.shape[1], self.train_X.shape[2]), return_sequences=True, dropout=drp))
        model.add(LSTM(n, return_sequences=True, dropout=drp))
        model.add(LSTM(n, return_sequences=True, dropout=drp))
        model.add(LSTM(n, return_sequences=True, dropout=drp))
        model.add(LSTM(n, return_sequences=True, dropout=drp))
        # model.add(LSTM(n, return_sequences=True, dropout=drp))
        # model.add(LSTM(n, return_sequences=True, dropout=drp))
        # model.add(LSTM(n, return_sequences=True, dropout=drp))
        # model.add(LSTM(n, return_sequences=True, dropout=drp))
        # model.add(LSTM(n, return_sequences=True, dropout=drp))
        # model.add(LSTM(n, return_sequences=True, dropout=drp))
        model.add(LSTM(n, dropout=drp))
        model.add(Dense(n))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Dropout(drp))
        # model.add(Dense(n, activation="relu"))
        # model.add(Dropout(drp))
        # model.add(Dense(n, activation="tanh"))
        # model.add(Dropout(drp))
        # model.add(Dense(44))
        # model.add(Dense(15, activation="tanh"))
        # model.add(Dropout(drp))
        model.add(Dense(100))
        model.add(Dense(1))
        model.compile(loss=Huber(delta=10.0), optimizer='adamax')
        # fit network
        # reshape(self.train_X, (self.n_batch, self.train_X.shape[0], self.train_X.shape[1], self.train_X.shape[2]))
        history = model.fit(self.train_X, self.train_y, epochs=self.epochs, batch_size=self.n_batch,
                            validation_data=(self.test_X, self.test_y), verbose=2,
                            shuffle=True)
        model.save("C:/Users/markk/Desktop/Brandywine/Model.h5")
        return model, history

    def plot_history(self):

        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    def run(self):

        # make a prediction
        yhat = self.model.predict(self.test_X)
        # invert scaling for forecast
        yhat_list = []
        for item in yhat:
            new = zeros(self.dataset.shape[1])
            new[0] = item[0]
            yhat_list.append(new)
        inv_yhat = yhat_list
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]
        trend_yhat = []
        ref_yhat = self.ref_data["Gage_Height"].values[(self.n_train_hours + self.s_length - 45):].copy()
        for x in range(len(ref_yhat) - 45):
            interval = ref_yhat[x: x + 45]
            interval -= detrend(interval)
            slope = interval[29] - interval[28]
            trend_yhat.append(interval[44] + slope)
        inv_yhat += trend_yhat

        # invert scaling for actual
        # test_y = self.test_y.reshape((len(self.test_y), 1))
        # y_list = []
        # for item in test_y:
        #     new = zeros(self.dataset.shape[1])
        #     new[0] = item[0]
        #     y_list.append(new)
        # inv_y = y_list
        # inv_y = self.scaler.inverse_transform(inv_y)
        # inv_y = inv_y[:, 0]
        inv_y = self.ref_data["Gage_Height"][self.n_train_hours + self.s_length:].values
        # inv_y = inv_y + self.ref_data["Gage_Height"][self.n_train_hours:-self.s_length]
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.4f' % rmse)

        pyplot.plot(inv_y, label='train')
        pyplot.plot(inv_yhat, label='test')
        pyplot.legend()
        pyplot.xlabel("Days")
        pyplot.ylabel("Water Level")
        pyplot.title("1-Step")
        pyplot.show()

    def transform(self, raw, inv=True, pos=0):
        if self.topf:
            to_trans = zeros(self.feat_no + 1)
        else:
            to_trans = zeros(self.feat_no)
        to_trans[pos] = raw
        to_trans = to_trans.reshape(1, len(to_trans))
        if inv:
            transform = self.scaler.inverse_transform(to_trans)
        else:
            transform = self.scaler.transform(to_trans)
        transform = transform[0][pos]
        return transform





if __name__ == "__main__":

    PATH = "/home/markkhusidman/Desktop/Brandywine/"
    r_PATH = PATH + "Brandywine_ref.csv"
    d_PATH = PATH + "Brandywine_test_EB2.csv"
    # r_PATH = PATH + "Brandywine_GHO_lr.csv"
    # d_PATH = PATH + "Brandywine_GHO_lr.csv"
    # r_PATH = PATH + "01115170_clean_diff.csv"
    # d_PATH = PATH + "01115170_clean_diff.csv"
    # r_PATH = PATH + "01115170_clean_bpass_scale_sub_c.csv"
    # d_PATH = PATH + "01115170_clean_bpass_diff_scale_sub_c.csv"
    # r_PATH = PATH + "01115170_clean_scale.csv"
    # d_PATH = PATH + "01115170_clean_diff_scale.csv"
    cols = [0, 1, 2, 3, 4, 5, 6, 7]
    m = Model(r_PATH, d_PATH, 4, 1800, 30, 150, 1, cols, diff=True, deriv=False)
    m.plot_history()
    m.run()
    # print(m.forecasts[0:11])


a = "01434498"
b = "01104430"
n = 0.0052083
z = 35136