from matplotlib import style
from StockClass import StockData, location
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance
import numpy as np
import time
import os
import csv

style.use('ggplot')
     
######################################################################################################  

class Game:

    def __init__(self):
        
        self.name = ''
        self.minute = -1
        self.allStocks = {}
        self.portfolio = {}
        self.money = 1000.00 # $$$
        
    def load_stocks(self, typeOfStocksFolder):

        directory = os.fsencode(location + typeOfStocksFolder)

        for file in os.listdir(directory):

            filename = typeOfStocksFolder + '/' + os.fsdecode(file)

            with open(filename) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                ticker = ''
        
                line_count = 0
                for row in csv_reader:

                    # Create the StockData instance
                    if (line_count == 0):
                        ticker = row[0]
                        self.allStocks[ticker] = StockData(ticker, False, typeOfStocksFolder)
                        self.allStocks[ticker].dataPerTime['date'] = []
                        self.allStocks[ticker].dataPerTime['open'] = []
                        self.allStocks[ticker].dataPerTime['high'] = []
                        self.allStocks[ticker].dataPerTime['low'] = []
                        self.allStocks[ticker].dataPerTime['close'] = []
                        self.allStocks[ticker].dataPerTime['volume'] = []
                    
                    elif (line_count > 1):
                        self.allStocks[ticker].dataPerTime['date'] += [row[0]]
                        self.allStocks[ticker].dataPerTime['open'] += [float(row[1])]
                        self.allStocks[ticker].dataPerTime['high'] += [float(row[2])]
                        self.allStocks[ticker].dataPerTime['low'] += [float(row[3])]
                        self.allStocks[ticker].dataPerTime['close'] += [float(row[4])]
                        self.allStocks[ticker].dataPerTime['volume'] += [float(row[5])]

                    line_count += 1

    def update_data(self):

        # I should add a feature that only adds data that does not exist, so I can run it at any time
        timer = 0
        for stock in self.allStocks:
            print(stock)
            if ((timer + 1) % 5 == 0):
                print('Hold Up! Gotta slow down because alpha vantage is a slimy BITCH!')
                print("Waiting[", end="")

                for i in range(61):
                    time.sleep(1)
                    print("=", end="")

                print("] DONE")

            # Find Last Date
            csvFileread = open(location + self.allStocks[stock].type + '/' + self.allStocks[stock].ticker + '.csv', 'r', newline='')
            reader = csv.reader(csvFileread)
            lastDate = ''
            for row in reader:
                lastDate = row[0]
            print(lastDate)
            csvFileread.close()

            # Add Data Past Last Date
            csvFileWrite = open(location + self.allStocks[stock].type + '/' + self.allStocks[stock].ticker + '.csv', 'a', newline='')
            writer = csv.writer(csvFileWrite)
            tempData = StockData(stock, False, self.allStocks[stock].type)
            tempData.install_data()

            counter = 0
            for date in tempData.dataPerTime['date']:
                if (str(date) == str(lastDate)):
                    break
                else:
                    counter += 1

            newRows = []
            indexAfterMatch = counter + 1
            for i in range(len(tempData.dataPerTime['date'][indexAfterMatch:-1])):
                newRows.append([tempData.dataPerTime['date'][indexAfterMatch + i], tempData.dataPerTime['open'][indexAfterMatch + i], tempData.dataPerTime['high'][indexAfterMatch + i], tempData.dataPerTime['low'][indexAfterMatch + i], tempData.dataPerTime['close'][indexAfterMatch + i], tempData.dataPerTime['volume'][indexAfterMatch + i]])

            writer.writerows(newRows)
            csvFileWrite.close()
            timer += 1

    def update_current_prices(self):
        for stock in self.allStocks:
            self.allStocks[stock].update_current_price(self.minute)
 
    def menu(self):

        print('')
        print('minute =', self.minute)
        print("Money = $", self.money)
        print('Portfolio :', self.portfolio)
        print('')

        for stock in self.allStocks:
            print(self.allStocks[stock].ticker, ":", self.allStocks[stock].currentPrice)

        print('')
        print('1. Buy')
        print('2. Sell')
        print('3. Next minute')
        print('4. QUIT')
        print('')
        
    def next_minute(self):
        self.minute += 1
        self.update_current_prices()
        self.menu()

    def action(self, playerInput):
        
        if (playerInput == '1'):
            self.purchase_stock()
            
        if (playerInput == '2'):
            self.sell_stock()
            
        if (playerInput == '3'):
            self.next_day()

        else:
            print("Invalid Input")

    def purchase_stock(self):
        
        stock = input("Which Stock?")
        
        bool = False
        for key in self.allStocks:
            
            if(stock == key):
                bool = True
                shares = input("How Many Shares")
                cost = float(shares) * float(self.allStocks[stock].currentPrice)
            
                if cost <= self.money:
                    
                    print("It Will Cost $", cost)
                    check = input('Are You Sure? Y/N')
        
                    if (check == 'Y' or check == 'y'):
                        self.money -= cost

                        found = False
                        for ticker in self.portfolio:
                            if (ticker == stock):
                                  found = True
                                  self.portfolio[stock][0] += float(shares)
                                  self.portfolio[stock][1] = float(self.allStocks[stock].currentPrice)

                        if (found == False):
                            self.portfolio.update({stock:[float(shares), float(self.allStocks[stock].currentPrice)]})

                else:
                    print('You Do Not Have Enough Money')
                    
                self.menu()
                break
            
        if (bool == False):
            print('That Stock Does Not Exist')
        
    def sell_stock(self):
        
        stock = input("Which Stock?")
        
        found = False
        for ticker in self.portfolio:
            
            if (stock == ticker):
                found = True
                shares = input("How Many Shares")
                
                if (float(shares) <= float(self.portfolio[stock][0])):
                    revenue = float(shares) * float(self.allStocks[stock].currentPrice)
                    print("You Will Add $", revenue)
                    check = input('Are You Sure? Y/N')
        
                    if (check == 'Y' or check == 'y'):
                        self.portfolio[stock][0] -= float(shares)
                        self.money += revenue
                    
                        if (float(self.portfolio[stock][0]) == 0):
                             self.portfolio.pop(stock)
                        self.menu() 
                                     
                else:
                    print('You Do Not Have That Many Shares of', stock)
                 
                break

        if (found == False):
            print('You Do Not Own That Stock!')

######################################################################################################
