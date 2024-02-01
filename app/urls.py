from django.urls import path
from . import views

urlpatterns = [
    path('', views.myfirst, name='myfirst'),
    path('process_by_id/', views.process_by_id, name='process_by_id'),
    path('process_by_smile/', views.process_by_smile, name='process_by_smile'),
    path('process_by_excel/', views.process_by_excel, name='process_by_excel'),
    path('download_excel/', views.download_excel, name='download_excel'),
]