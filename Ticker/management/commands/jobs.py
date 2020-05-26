from django.core.management.base import BaseCommand
from .webAPI import get_data, manage_metadata
from .indexs import addIndicators, manage_indicators


class Command(BaseCommand):

    def handle(self, symbol='IBM', *args, **options):
        manage_metadata(get_data(symbol))
        manage_indicators(addIndicators(symbol), symbol)
        print('refreshed ...')

