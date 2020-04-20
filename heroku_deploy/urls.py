from django.urls import path, include
from heroku_deploy.views import index, chatbot
from django.conf.urls import url

urlpatterns = [
    # url(r'', index, name='index'),
    url(r'chatbot', chatbot, name='chatbot')
]
