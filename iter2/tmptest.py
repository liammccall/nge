import yfinance as yf
import pandas as pd
all_data : pd.DataFrame = yf.download("SPY", period="2y", progress=False)
print(all_data.columns)

def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()

from backtesting import Strategy
from backtesting.lib import crossover


class SmaCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 20
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()
            
    
    # def next(self):
    #     if (self.sma1[-2] < self.sma2[-2] and
    #             self.sma1[-1] > self.sma2[-1]):
    #         self.position.close()
    #         self.buy()

    #     elif (self.sma1[-2] > self.sma2[-2] and    # Ugh!
    #           self.sma1[-1] < self.sma2[-1]):
    #         self.position.close()
    #         self.sell()
    
import backtesting
from backtesting import Backtest
from backtesting.test import GOOG

all_data.columns = [column[0] for column in all_data.columns]

# print(GOOG.columns)

# print(all_data.columns)

bt = Backtest(all_data, SmaCross, cash=10_000, commission=.002)
stats = bt.run()
print(stats["Start"])

# print(backtesting.test())