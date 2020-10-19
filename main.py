from GameClass import Game
from StockClass import StockData

######################################################################################################  

def download_everything(tickers, boolCSV, type):
    
    print(len(tickers), 'Tickers')

    # Temp 75, finish installing this shit
    for i in range(len(tickers)):

        thisStock = StockData(tickers[i], boolCSV, type)
        thisStock.install_data()

        if ((i + 1) % 5 == 0):
            print('Hold Up! Gotta slow down because alpha vantage is a slimy BITCH!')
            print("Waiting[", end="")

            for i in range(61):
                time.sleep(1)
                print("=", end="")

            print("] DONE")

def play_game():

    theGame = Game()
    theGame.name = input("What is your name? ")
    theGame.load_stocks(input("What type of Stocks? "))
    theGame.next_day()

    print('What would you like to do', theGame.name, "?")
    playerInput = input()

    while(playerInput != '4'):
        theGame.action(playerInput)
        print('What would you like to do', theGame.name, "?")
        playerInput = input()

######################################################################################################  