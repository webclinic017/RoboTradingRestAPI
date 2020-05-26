from rest_framework import serializers
from .models import Ticker, StockMetaData, Indicator


class IndicatorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Indicator
        fields = '__all__'


class TickerSerializer(serializers.ModelSerializer):
    # stock = serializers.StringRelatedField(many=False, read_only=True, allow_null=True)
    indicators = IndicatorSerializer(many=True, read_only=True)

    class Meta:
        model = Ticker
        fields = '__all__'  # ['date', 'open', 'high', 'low', 'close', 'volume', 'indicators']


class StockMetaDataSerializer(serializers.ModelSerializer):
    ticker = TickerSerializer(many=True, read_only=True)
    #indicators = IndicatorSerializer(many=True, read_only=True)

    class Meta:
        model = StockMetaData
        fields = ['Symbol', 'Interval', 'Last_Refreshed', 'ticker']
