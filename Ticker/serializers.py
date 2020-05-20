from rest_framework import serializers
from .models import Ticker, StockMetaData


class TickerSerializer(serializers.ModelSerializer):
    # stock = serializers.StringRelatedField(many=False, read_only=True, allow_null=True)

    class Meta:
        model = Ticker
        fields = ['date', 'open', 'high', 'low', 'close', 'volume']


class StockMetaDataSerializer(serializers.ModelSerializer):
    ticker = TickerSerializer(many=True, read_only=True)

    class Meta:
        model = StockMetaData
        fields = ['Symbol', 'Interval', 'Last_Refreshed', 'ticker']