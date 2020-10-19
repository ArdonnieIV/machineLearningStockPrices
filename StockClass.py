from alpha_vantage.timeseries import TimeSeries
import csv
import os

location = 'C:/MyCode/machineLearningStockPrices/' # TODO : Change location to the address of your reposity.

######################################################################################################  

class StockData:

    def __init__(self, tickerInput, toCreateCSV, typeOfStock):
        
        self.ticker = tickerInput
        self.dataPerTime = {}
        self.type = typeOfStock
        self.createCSV = toCreateCSV
        self.currentPrice = 0
        self.linearRegressionAc = 0
    
    def install_data(self):
        
        ts = TimeSeries(key='2NJ3T8RDHYZJVIG5', output_format='pandas')
        data, meta_data = ts.get_intraday(self.ticker, interval='1min', outputsize='full')

        self.dataPerTime['date'] = data['1. open'].index[::-1]
        self.dataPerTime['open'] = data['1. open'][::-1]       
        self.dataPerTime['high'] = data['2. high'][::-1]
        self.dataPerTime['low'] = data['3. low'][::-1]
        self.dataPerTime['close'] = data['4. close'][::-1]
        self.dataPerTime['volume'] = data['5. volume'][::-1]

        if self.createCSV:
            self.create_csv_file()

    def get_minimum_value(self, category):

        minimum = float(self.dataPerTime[category][0])
        for value in self.dataPerTime[category]:
            if float(value) < minimum:
                minimum = float(value)

        return minimum

    def get_maximum_value(self, category):

        maximum = float(self.dataPerTime[category][0])
        for value in self.dataPerTime[category]:
            if float(value) > maximum:
                maximum = float(value)

        return maximum

    def read_data_from_csv(self):

        fileName = os.fsencode(location + self.type + '/' + self.ticker + '.csv')
        with open(fileName) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            row_counter = 0

            for row in csv_reader:
                if (row_counter == 0):
                    self.dataPerTime['date'] = []
                    self.dataPerTime['open'] = []
                    self.dataPerTime['high'] = []
                    self.dataPerTime['low'] = []
                    self.dataPerTime['close'] = []
                    self.dataPerTime['volume'] = []

                elif (row_counter > 1):
                    self.dataPerTime['date'] += [row[0]]
                    self.dataPerTime['open'] += [float(row[1])]
                    self.dataPerTime['high'] += [float(row[2])]
                    self.dataPerTime['low'] += [float(row[3])]
                    self.dataPerTime['close'] += [float(row[4])]
                    self.dataPerTime['volume'] += [float(row[5])]

                row_counter += 1
    
    def update_current_price(self, minute):

        self.currentPrice = self.dataPerTime['close'][minute]
    
    def print_data(self):
        print(self.ticker)
        print('Date, Open, High, Low, Close, Volume')
        for i in range(len(self.dataPerTime['date'])):
            print(self.dataPerTime['date'][i],end=',')
            print(self.dataPerTime['open'][i],end=',')
            print(self.dataPerTime['high'][i],end=',')
            print(self.dataPerTime['low'][i],end=',')
            print(self.dataPerTime['close'][i],end=',')
            print(self.dataPerTime['volume'][i])

    def create_csv_file(self):
        fileName = self.ticker + '.csv'
        print(fileName)
       
        with open(fileName, 'w', newline='') as file:
            a = csv.writer(file, delimiter=',')
            a.writerow([self.ticker])
            a.writerow(['time', 'open', 'high', 'low', 'close', 'volume'])

            dataSize = len(self.dataPerTime['date'])
            print(dataSize, 'points of data')
            dataSize -= 1
            
            for i in range(dataSize):
                    a.writerow([self.dataPerTime['date'][dataSize-i], self.dataPerTime['open'][dataSize-i], self.dataPerTime['high'][dataSize-i], self.dataPerTime['low'][dataSize-i], self.dataPerTime['close'][dataSize-i], self.dataPerTime['volume'][dataSize-i]])
                   
        os.rename(location + fileName, location + self.type + '/' + fileName)

######################################################################################################  
