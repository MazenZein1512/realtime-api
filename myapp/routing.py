from django.urls import re_path
from . import consumers  # Make sure you have a consumer for WebSocket handling

# Define the WebSocket URL pattern
websocket_urlpatterns = [
    re_path(r'ws/audio/$', consumers.AudioConsumer.as_asgi()),  # Ensure this matches your URL
]
