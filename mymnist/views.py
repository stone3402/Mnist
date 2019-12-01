from django.http import HttpResponse
from django.shortcuts import render

from .models import conver_to_mnist, predict, save_all_mnist


def index(request):
    return HttpResponse("hello django  !")


def detail(request, num):
    return HttpResponse("detail-%s" % num)


def mnist(request):
    result = "ng"

    return render(request, 'mymnist/mnist.html', {'resutl': result})


def upload(request):
    # save_all_mnist()

    convas_jpg = request.POST['canvasData']

    mnist_data = conver_to_mnist(convas_jpg)

    score, result = predict(mnist_data)

    return render(request, 'mymnist/mnist.html', {'score': score, 'result': result})
