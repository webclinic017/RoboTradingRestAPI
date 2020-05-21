from django.contrib import admin
from .models import StockMetaData, Ticker, Indicator
# Register your models here.

admin.site.register(StockMetaData)
admin.site.register(Ticker)
admin.site.register(Indicator)