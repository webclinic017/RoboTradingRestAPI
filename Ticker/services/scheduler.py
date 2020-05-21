import schedule
import time
from .AlphaVantage import get_data, manage_metadata
from .indicators import addIndicators, manage_indicators


def scheduler():
    def dataManager():
        data = get_data()
        manage_metadata(data)
        new_data = addIndicators()
        manage_indicators(new_data)
        print('refreshed ...')

    schedule.every(10).seconds.do(dataManager)

    while True:
        schedule.run_pending()
        time.sleep(1)

