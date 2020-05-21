from django.shortcuts import render
from .models import StockMetaData, Ticker
from .serializers import StockMetaDataSerializer
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from .services.scheduler import scheduler
from django_simple_task import defer
import time
import asyncio
# Create your views here.

'''
def task1():
    time.sleep(1)
    print("task1 done")

async def task2():
    await asyncio.sleep(1)
    print("task2 done")'''

def stock_list(request):
    #scheduler()
    #defer(task1)
    #defer(task2)
    if request.method == 'GET':
        stocks = StockMetaData.objects.all()
        serializer = StockMetaDataSerializer(stocks, many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = StockMetaDataSerializer(data=data)

        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        return JsonResponse(serializer.errors, status=400)
