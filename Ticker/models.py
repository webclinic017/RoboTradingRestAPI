from django.db import models

# Create your models here.


class StockMetaData(models.Model):
    Symbol = models.CharField(max_length=32)  # the stock name
    Interval = models.CharField(max_length=32)  # data interval
    Last_Refreshed = models.DateTimeField()  # refresh date

    def __str__(self):
        return str(self.Symbol)


class Ticker(models.Model):
    date = models.DateTimeField()  # date
    stock = models.ForeignKey(StockMetaData, on_delete=models.CASCADE, related_name='ticker')
    open = models.FloatField()  # Opening Stock price
    high = models.FloatField()  # highest Stock price
    low = models.FloatField()  # lowest Stock price
    close = models.FloatField()  # closing of the stock price
    volume = models.FloatField()  # How many times sales was done for the particular stock

    def __str__(self):
        return str(self.date)


class Indicator(models.Model):
    ticker = models.ForeignKey(
        Ticker,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name='indicators'
    )
    date = models.DateTimeField()  # date
    RSI_14D = models.FloatField()
    Volume_plain = models.FloatField()
    BB_Middle_Band = models.FloatField()
    BB_Upper_Band = models.FloatField()
    BB_Lower_Band = models.FloatField()
    Aroon_Oscillator = models.FloatField()
    PVT = models.FloatField()
    AB_Middle_Band = models.FloatField()
    AB_Upper_Band = models.FloatField()
    AB_Lower_Band = models.FloatField()
    STOK = models.FloatField()
    STOD = models.FloatField()
    Chaikin_MF = models.FloatField()
    psar = models.FloatField()
    ROC = models.FloatField()
    VWAP = models.FloatField()
    Momentum = models.FloatField()
    CCI = models.FloatField()
    OBV = models.FloatField()
    Kelch_Upper = models.FloatField()
    Kelch_Middle = models.FloatField()
    Kelch_Down = models.FloatField()
    TEMA = models.FloatField()
    NATR = models.FloatField()
    plusDI = models.FloatField()
    minusDI = models.FloatField()
    ADX = models.FloatField()
    MACD = models.FloatField()
    Money_Flow_Index = models.FloatField()
    turning_line = models.FloatField()
    standard_line = models.FloatField()
    ichimoku_span1 = models.FloatField()
    ichimoku_span2 = models.FloatField()
    chikou_span = models.FloatField()
    WillR = models.FloatField()
    MIN_Volume = models.FloatField()
    MAX_Volume = models.FloatField()
    KAMA = models.FloatField()

    def __str__(self):
        return str(self.date) # str(self.stock) + str(self.date)

