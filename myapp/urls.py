from django.urls import path
from . import views
from django.views.generic import TemplateView

urlpatterns = [
    path('', views.index, name='index'),
    path('favicon.ico', TemplateView.as_view(template_name='myapp/favicon.ico', content_type='image/x-icon'))
]