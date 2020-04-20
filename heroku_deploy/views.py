from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import redirect

from heroku_deploy.chatbot import activate_bot
# Create your views here.
import json
# import MySQLdb #-------- use "%pip install mysqlclient" to install
# import pandas as pd

# mydb = MySQLdb.connect(host='localhost',user='root',password='root',db='bot')
# cursor = mydb.cursor()

def index(request):
    return HttpResponse("<h1>Sample Project</h1>")

def chatbot(request):
    msg = ''
    rpl = ''

    print("chatbot")
    if(request.method == "POST"):
        print("chatbot_post")
        msg = request.POST.get('query')
        print(msg)
        rpl = activate_bot(msg)
        print(rpl)
        # tr = "INSERT INTO message(msg)VALUES("+str(rpl)+");"
        # cursor.execute(tr)
        # mydb.commit()
        # cursor.close()
        # redirect('chatbot')
        # return render(request, "login_css.html", {"text":msg})
        # return render_template()
    else:
        rpl = "Welcome to the Tacto chat bot!"
    return render(request, "login_css.html", {"message":rpl})