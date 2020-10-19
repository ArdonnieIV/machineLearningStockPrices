import pandas as pd
import math, datetime
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from GameClass import Game
from StockClass import StockData, location
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import style
from matplotlib import animation
from matplotlib.pylab import *
#from mpl_toolkits.axes_grid1 import host_subplot
#from keras.layers.core import Dense, Activation, Dropout
#from keras.layers.recurrent import LSTM
#from keras.models import Sequential
import time

style.use('ggplot')

######################################################################################################  

def display_data(type): # Just for visual practice

    sp500Game = Game()
    sp500Game.load_stocks(type)

    numOfGraphs = 6
    i = 1
    for stock in sp500Game.allStocks:
        daMean = []
        counter = 1

        meanRange = 500
        for minute in sp500Game.allStocks[stock].dataPerTime['close']:
            if counter >= meanRange:
                daMean.append(stats.mean(sp500Game.allStocks[stock].dataPerTime['close'][counter - meanRange:counter]))
                counter += 1

            else:
                daMean.append(stats.mean(sp500Game.allStocks[stock].dataPerTime['close'][0:counter]))
                counter += 1

        xs = np.array(list(range(len(sp500Game.allStocks[stock].dataPerTime['close']))), dtype=np.float64)
        ys = np.array(sp500Game.allStocks[stock].dataPerTime['close'], dtype=np.float64)
        m = best_fit_slope(xs,ys)
        b = best_fit_intercept(xs,ys,m)
        bestFitLine = [(m*x)+b for x in xs]

        plt.figure(i)
        plt.plot(bestFitLine, 'b')
        plt.plot(sp500Game.allStocks[stock].dataPerTime['close'], 'r')
        plt.plot(daMean, 'b')
        plt.ylabel('Price')
        plt.xlabel('Minutes')
        plt.title(stock + " " + str(int(stats.mean(sp500Game.allStocks[stock].dataPerTime['volume']))))
    
        if i == numOfGraphs:
            plt.show()
            i = 1

        else:
            i += 1

def best_fit_slope(xs,ys):

    m = (((stats.mean(xs)*stats.mean(ys)) - stats.mean(xs*ys)) /
         ((stats.mean(xs)*stats.mean(xs)) - stats.mean(xs*xs)))

    return m

def best_fit_intercept(xs,ys,m):

    return stats.mean(ys) - m*stats.mean(xs)

def slope(leftX, leftY, rightX, rightY):
          return float(rightY - leftY) / float(rightX - leftX)

def derivative_of(aList):
    derivative = []
    for i in range(len(aList)):

        # Special Formula for first and last elements
        if (i == 0):
            derivative += [float(aList[i + 1]) - float(aList[i])]
        elif (i == len(aList) - 1):
            derivative += [float(aList[i]) - float(aList[i - 1])]

        else:
            derivative += [(float(aList[i + 1]) - float(aList[i - 1])) / 2]

    return derivative
    
def correlate_all_data():
    sp500Game = Game()
    sp500Game.load_stocks('under10')

    averagePrices = np.array([stats.mean(sp500Game.allStocks[stock].dataPerTime['close']) for stock in sp500Game.allStocks], dtype=np.float64)
    averageVolumes = np.array([stats.mean(sp500Game.allStocks[stock].dataPerTime['volume']) for stock in sp500Game.allStocks], dtype=np.float64)
    bestFitSlopes = np.array([best_fit_slope(np.array(list(range(len(sp500Game.allStocks[stock].dataPerTime['close']))), dtype=np.float64), np.array(sp500Game.allStocks[stock].dataPerTime['close'], dtype=np.float64)) for stock in sp500Game.allStocks], dtype=np.float64)

    print(np.corrcoef(averagePrices, averageVolumes))
    print(np.corrcoef(averagePrices, bestFitSlopes))
    print(np.corrcoef(averageVolumes, bestFitSlopes))
    
def keras_train_classifier(data, model, input_forcast_out):

    data_frame = pd.DataFrame(data)
    data_frame = data_frame[['date', 'high', 'low', 'open', 'close', 'volume']]
    data_frame['high-low-percent'] = (data_frame['high'] - data_frame['close']) / data_frame['close'] * 100
    data_frame['percent-change'] = (data_frame['close'] - data_frame['open']) / data_frame['open'] * 100
    data_frame = data_frame[['close', 'high-low-percent', 'percent-change', 'volume']]
    data_frame.fillna(-99999, inplace=True)

    data_frame['label'] = data_frame['close'].shift(-input_forcast_out)

    X = np.array(data_frame.drop(['label'], 1))
    X_lately = X[-input_forcast_out:]
    X = X[0:-input_forcast_out]
   
    data_frame.dropna(inplace=True)
    y = np.array(data_frame['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    X_lately = np.reshape(X_lately, (X_lately.shape[0], 1, X_lately.shape[1]))

    model.add(LSTM(32, input_shape=(None, 4), return_sequences = True))
    model.add(LSTM(16))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer='adagrad')
    model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=2)

    predictions = list(model.predict(X)) + list(model.predict(X_lately))
    return predictions

def train_classifier(data, classifier, input_forcast_out):

    data_frame = pd.DataFrame(data)
    data_frame = data_frame[['date', 'high', 'low', 'open', 'close', 'volume']]
    data_frame['high-low-percent'] = (data_frame['high'] - data_frame['close']) / data_frame['close'] * 100
    data_frame['percent-change'] = (data_frame['close'] - data_frame['open']) / data_frame['open'] * 100
    data_frame = data_frame[['close', 'high-low-percent', 'percent-change', 'volume']]
    data_frame.fillna(-99999, inplace=True)

    data_frame['label'] = data_frame['close'].shift(-input_forcast_out)

    X = np.array(data_frame.drop(['label'], 1))
    X_lately = X[-input_forcast_out:]
    X = X[0:-input_forcast_out]
   
    data_frame.dropna(inplace=True)
    y = np.array(data_frame['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

    classifier.fit(X_train, y_train)
    allPrediction = list(classifier.predict(X)) + list(classifier.predict(X_lately))
    
    return allPrediction
    
def test_train_classifier(data, classifier, input_forcast_out, range):

    # possibly change the X variable, instead of one instance of data, it could be a list of instances
    data_frame = pd.DataFrame(data)
    data_frame = data_frame[['date', 'high', 'low', 'open', 'close', 'volume']]
    data_frame['high-low-percent'] = (data_frame['high'] - data_frame['close']) / data_frame['close'] * 100
    data_frame['percent-change'] = (data_frame['close'] - data_frame['open']) / data_frame['open'] * 100
    data_frame = data_frame[['close', 'high-low-percent', 'percent-change', 'volume']]
    data_frame.fillna(-99999, inplace=True)

    data_frame['label'] = data_frame['close'].shift(-input_forcast_out)

    X = np.array(data_frame.drop(['label'], 1))
    X_lately = X[-input_forcast_out:]
    X = X[0:-input_forcast_out]
   
    data_frame.dropna(inplace=True)
    y = np.array(data_frame['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

    classifier.fit(X_train, y_train)
    allPrediction = list(classifier.predict(X)) + list(classifier.predict(X_lately))

    plt.plot(allPrediction, 'b')
    plt.plot(data['close'], 'r')
    plt.show()

    return allPrediction

def simple_predict(data, classifier, range):

    limited_dataPerTime = {}
    for key in data:
        limited_dataPerTime[key] = data[key][0:range]

    data_frame = pd.DataFrame(limited_dataPerTime)
    data_frame = data_frame[['date', 'high', 'low', 'open', 'close', 'volume']]
    data_frame['high-low-percent'] = (data_frame['high'] - data_frame['close']) / data_frame['close'] * 100
    data_frame['percent-change'] = (data_frame['close'] - data_frame['open']) / data_frame['open'] * 100
    data_frame = data_frame[['close', 'high-low-percent', 'percent-change', 'volume']]
    data_frame.fillna(-99999, inplace=True)
    data_frame.dropna(inplace=True)

    return classifier.predict(data_frame)

def limit_train(inputStock, range, classifier, forcast_out):

    limited_dataPerTime = inputStock.dataPerTime
    for key in limited_dataPerTime:
        limited_dataPerTime[key] = limited_dataPerTime[key][:range]

    return train_classifier(limited_dataPerTime, classifier, forcast_out)

def live_graph(inputStock, classifier, forcast_out):

    def updateData(i):

        startRange = 100 # Start with a 100 pieces of data
        dataRange = startRange + i
        # Black
        realPrices = inputStock.dataPerTime['close'][:dataRange]
        x1 = list(range(dataRange))
        
        # Red
        prediction = test_train_classifier(inputStock.dataPerTime, classifier, forcast_out, dataRange)
        
        p011.set_data(x1,realPrices)
        p012.set_data(x1,prediction)

        ax01.set_xlim(i, 155 + i)

        return p011, p012

    # Sent for figure
    font = {'size': 9}
    matplotlib.rc('font', **font)

    # Setup figure and subplots
    f0 = figure(num = 0, figsize = (12, 8))
    f0.suptitle(inputStock.ticker + " Stock Price", fontsize=12)
    ax01 = subplot2grid((2, 2), (0, 0))

    # Set titles of subplots
    ax01.set_title('Price vs Time')

    # set y-limits
    RealtopY = inputStock.get_maximum_value('close')
    RealbottomY = inputStock.get_minimum_value('close')
    ax01.set_ylim(RealbottomY,RealtopY)
    
    # set first x-limits
    ax01.set_xlim(0,155)

    # Turn on grids
    ax01.grid(True)

    # set label names
    ax01.set_xlabel("minute")
    ax01.set_ylabel("price")

    # set plots
    p011, = ax01.plot('b-', label="Real")
    p012, = ax01.plot('r-', label="Predicted")

    simulation = animation.FuncAnimation(f0, updateData, blit=False, frames=len(inputStock.dataPerTime['close']), interval=1, repeat=False)
    plt.show()

def find_low_points(data):

    derivatives = derivative_of(data.dataPerTime['close'])
    x = []
    y = []

    for minute in range(len(data.dataPerTime['date']) - 1):
        if (minute is not 0) and (minute is not (len(data.dataPerTime['date']) - 1)):
            if (derivatives[minute - 1] < 0) and (derivatives[minute + 1] > 0): # If Slope goes from negative to positive then it is a local min
                x.append(minute)
                y.append(data.dataPerTime['close'][minute])

    return x, y

def find_high_points(data):

    derivatives = derivative_of(data.dataPerTime['close'])
    x = []
    y = []

    for minute in range(len(data.dataPerTime['date']) - 1):
        if (minute is not 0) and (minute is not (len(data.dataPerTime['date']) - 1)):
            if (derivatives[minute - 1] > 0) and (derivatives[minute + 1] < 0): # If Slope goes from positive to negative then it is a local max
                x.append(minute)
                y.append(data.dataPerTime['close'][minute])

    return x, y

def lowest_ranged(xContainer, yContainer, data, width):

    halfRange = int(width / 2)
    x = []
    y = []

    dumb = len(yContainer)
    for index in range(dumb):
        if (index > halfRange) and (index < (dumb - halfRange)):
            if (yContainer[index] == min(yContainer[index - halfRange:index + halfRange])):
                x.append(xContainer[index])
                y.append(yContainer[index])
    
    i = 0
    while (i < (len(x) - 1)):
        leftX = x[i]
        leftY = y[i]
        rightX = x[i + 1]
        rightY = y[i + 1]
        M = slope(leftX, leftY, rightX, rightY)
        for j in range(rightX - leftX):
            linearPoint = leftY + (M*j)
            realPoint = data[leftX + j]
            if (realPoint < linearPoint):
                x.insert((i+1), (leftX + j))
                y.insert((i+1), realPoint)
                break
        i += 1
    
    return x, y

def highest_ranged(xContainer, yContainer, data, width):

    halfRange = int(width / 2)
    x = []
    y = []

    dumb = len(yContainer)
    for index in range(dumb):
        if (index > halfRange) and (index < (dumb - halfRange)):
            if (yContainer[index] == max(yContainer[index - halfRange:index + halfRange])):
                x.append(xContainer[index])
                y.append(yContainer[index])

    i = 0
    while (i < (len(x) - 1)):
        leftX = x[i]
        leftY = y[i]
        rightX = x[i + 1]
        rightY = y[i + 1]
        M = slope(leftX, leftY, rightX, rightY)
        for j in range(rightX - leftX):
            linearPoint = leftY + (M*j)
            realPoint = data[leftX + j]
            if (realPoint > linearPoint):
                x.insert((i+1), (leftX + j))
                y.insert((i+1), realPoint)
                break
        i += 1

    return x, y

######################################################################################################

#global myClassifier
#global kerasClassifier 

myClassifier = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)
#kerasClassifier = Sequential()

storage = Game()
storage.load_stocks('under10')
forcast_out = 50
testRange = 15

for stock in storage.allStocks:
    #predictions = keras_train_classifier(storage.allStocks[stock].dataPerTime, kerasClassifier, forcast_out)
    predictions = train_classifier(storage.allStocks[stock].dataPerTime, myClassifier, forcast_out)
    #plt.plot(predictions, 'b')

    plt.plot(storage.allStocks[stock].dataPerTime['close'], 'b')
    #x, y = find_low_points(storage.allStocks[stock])
    #j, k = find_high_points(storage.allStocks[stock])
    #x, y = lowest_ranged(x, y, storage.allStocks[stock].dataPerTime['close'], testRange)
    #j, k = highest_ranged(j, k, storage.allStocks[stock].dataPerTime['close'], testRange)
    #plt.plot(x, y, 'r')
    #plt.plot(j, k, 'y')
    plt.plot(predictions, 'r')
    plt.title(stock)
    plt.show()
