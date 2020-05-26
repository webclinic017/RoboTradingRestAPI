from django.shortcuts import render
from .models import StockMetaData, Ticker
from .serializers import StockMetaDataSerializer
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from .services.scheduler import  scheduled_jobs
# Create your views here.


def stock_list(request):
    scheduled_jobs()
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
