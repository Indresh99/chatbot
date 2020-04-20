import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json
from bs4 import BeautifulSoup
import requests
from io import BytesIO
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
from IPython.display import display 
import pandas as pd
import pickle
import os
from heroku.settings import BASE_DIR

stemmer = LancasterStemmer()

with open(BASE_DIR+ '/static/json/intents.json') as file:
    data = json.load(file)

# try:
#     with open('data.pickle', 'rb') as file:
#         words, labels, training, output = pickle.load(file)
#     print("data already present")
# except:
words = []
labels = []
docs_x = []
docs_y = []

for intents in data['intents']:
    for pattern in intents['patterns']:
#         print(pattern)
        wrd = nltk.word_tokenize(pattern)
#         print(wrd)
        words.extend(wrd)
        docs_x.append(wrd)
        docs_y.append(intents['tag'])
    
    if(intents['tag'] not in labels):
        labels.append(intents['tag'])

after_stemming = [stemmer.stem(w) for w in words if w not in "?"]
after_stemming = sorted(list(set(after_stemming)))
labels = sorted(labels)

trainig = []
output = []

out_empty = [0 for _ in range(len(labels))]

for i, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in after_stemming:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
            
    output_row = out_empty[:]
    output_row[labels.index(docs_y[i])] = 1
#     print(docs_y[i])
    
    trainig.append(bag)
    output.append(output_row)
    # with open('data.pickle', 'wb') as file:
    #     pickle.dump((words, labels, trainig, output), file)
    # print("data not present")


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(trainig[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
if (os.path.exists("model.tflearn" + ".meta")):
    model.load("model.tflearn")
    print("model present")
else:
    model.fit(trainig, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    print("model not present")

    

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w) for w in s_words]
    
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return numpy.array(bag)

def scrap():
    try:
        df =pd.read_csv("/Users/indresh/Desktop/src_links.csv")
        return random.choices(df["links"])[0]
#             print(random.choices(file)[0])
#         return random.choices(file)[0]
    # #     scale-with-grid
    except:
        URL = "https://www.tacto.in/team/"
        r = requests.get(URL) 

        soup = BeautifulSoup(r.content, 'html5lib') 
        # img = soup.find('div', attrs = {'id':'image_wrapper'}) 
        links = soup.find_all('img', attrs = {'class':'scale-with-grid'})
        # for row in img.find('img', attrs = {'class':'scale-with-grid'}): 
        #     print(row['src'])
        src_list = [links[link]["src"] for link in range(len(links))]
        df = pd.DataFrame(src_list, columns=["links"])
        df.to_csv("/Users/indresh/Desktop/src_links.csv", index=False)
    #         src_list_bytes = bytes(str(src_list), 'utf-8')
    #         with open("/Users/indresh/Desktop/src_links.txt", "wb") as links:
    #             links.write(src_list_bytes)
        return random.choices(src_list)[0]
        

def activate_bot(msg):
    while True:
        # inp = input("You: ")
        if msg.lower() == "q":
            break
        result = model.predict([bag_of_words(msg, after_stemming)])[0]
        print(result)
        result_index = numpy.argmax(result)
        tag = labels[result_index]
        print(result[result_index])
        if (result[result_index] > 0.7):
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            if(random.choice(responses) == "just_scrap"):
                link = scrap()
                print(link)
                response = requests.get(link)
                img = Image.open(BytesIO(response.content))
    #             imshow(np.asarray(img))
                display(img)
                return link
    #             print(img)
            else:
                return random.choice(responses)

        else:
            return "I didn't get it"