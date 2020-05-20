from django.urls import path
from .views import stock_list

urlpatterns = [
    path('ticker/', stock_list),
]
