import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from StockClass import StockData, location
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot


# interval: draw new frame every 'interval' ms
# frames: number of frames to draw

def live_graph_prediction(inputStock, forcast_out):

    def updateData(i):

        range = 100 + i # Start with a 100 pieces of data
        realPrices = inputStock.dataPerTime['close'][0:i]
        predictedPrices, testClassifier = limit_train(inputStock, range, testClassifier, forcast_out)
        t = list(range(i))

        p011.set_data(t,realPrices)
        p012.set_data((t + 50),predictedPrices)

        if i > 100:
            ax01.set_xlim(i - 100, i)

        return p011, p012,

    testClassifier = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1)

    # Sent for figure
    font = {'size'   : 9}
    matplotlib.rc('font', **font)

    # Setup figure and subplots
    f0 = figure(num = 0, figsize = (12, 8))#, dpi = 100)
    f0.suptitle(inputStock.ticker + " Stock Price", fontsize=12)
    ax01 = subplot2grid((2, 2), (0, 0))

    # Set titles of subplots
    ax01.set_title('Price vs Time')

    # set y-limits # Messy
    RealtopY = inputStock.get_maximum_value('close')
    RealbottomY = inputStock.get_minimum_value('close')
    PredicTopY = max(predictedData)
    PredicBottomY = min(predictedData)
    maxY = int(max(PredicTopY, RealtopY))
    minY = int(min(PredicBottomY, RealbottomY))
    ax01.set_ylim(minY,maxY)
    
    # set first x-limits
    ax01.set_xlim(0,100)

    # Turn on grids
    ax01.grid(True)

    # set label names
    ax01.set_xlabel("minute")
    ax01.set_ylabel("price")

    # set plots
    p011, = ax01.plot('b-', label="Real")
    p012, = ax01.plot('g-', label="Predicted")

    simulation = animation.FuncAnimation(f0, updateData, blit=False, frames=len(inputStock.dataPerTime['close']), interval=5, repeat=False)
    plt.show()

# Uncomment the next line if you want to save the animation
#simulation.save(filename='sim.mp4',fps=30,dpi=300)

