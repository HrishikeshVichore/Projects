"""Automated_Cart URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import path, include # new
from django.views.generic import RedirectView
from Automated_Cart.views import cart_items, checkout, sign_up_page, login_check, formView
from django.conf.urls import url

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', RedirectView.as_view(url='/check_login/', permanent=True)),
    path('accounts/', include('django.contrib.auth.urls')), # new
    url('signup/', sign_up_page),
    url('check_login/', formView),
    url('login_check/', login_check),
    url('cart/', cart_items),
    url('checkout/', checkout),
]