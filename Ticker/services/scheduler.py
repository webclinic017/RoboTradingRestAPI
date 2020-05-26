from .AlphaVantage import get_data, manage_metadata
from .indicators import addIndicators, manage_indicators


def scheduled_jobs(symbol='IBM'):
    manage_metadata(get_data(symbol))
    manage_indicators(addIndicators(symbol), symbol)
    print('refreshed ...')

