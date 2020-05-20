from django.shortcuts import render
from .models import StockMetaData
from .serializers import StockMetaDataSerializer
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from .services.alphaVintage import get_data, manage_metadata
# Create your views here.


def stock_list(request):

    if request.method == 'GET':
        data = get_data()
        manage_metadata(data)
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
