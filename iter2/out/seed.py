from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd


def SMA(values, n):
    return pd.Series(values).rolling(n).mean().values


class Model(Strategy):
    # MA parameters (can be optimized later)
    n1 = @n1
    n2 = @n2
    trend_n = @trend_n  # long-term filter

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
        self.trend = self.I(SMA, self.data.Close, self.trend_n)

    def next(self):
        price = self.data.Close[-1]
        trend = self.trend[-1]

        # -----------------------------
        # Trend filter (long-only bias)
        # -----------------------------
        uptrend = price > trend

        # -----------------------------
        # Entry / exit logic
        # -----------------------------
        cross_up = crossover(self.sma1, self.sma2)
        cross_down = crossover(self.sma2, self.sma1)

        # Enter long only in uptrend
        if cross_up and uptrend:
            self.position.close()
            self.buy()

        # Exit on bearish crossover
        elif cross_down:
            self.position.close()