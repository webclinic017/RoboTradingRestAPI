from django.contrib import admin
from .models import StockMetaData, Ticker
# Register your models here.

admin.site.register(StockMetaData)
admin.site.register(Ticker)