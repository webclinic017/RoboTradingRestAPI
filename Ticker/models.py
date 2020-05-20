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
    volume = models.IntegerField()  # How many times sales was done for the particular stock

    def __str__(self):
        return str(self.stock) + str(self.date)