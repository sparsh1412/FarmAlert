from django.shortcuts import render, redirect
from django.contrib import messages
from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup
from django.contrib.auth.decorators import login_required
import geocoder
from googleplaces import GooglePlaces, types, lang
import googlemaps
import json as simplejson
from services.forms import SoilForm
import pandas as pd
from sklearn.externals import joblib 

# Create your views here.


@login_required
def serve(request):
    return render(request, 'services/farmServices.html')

def PredNN(Input):
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder 


    # In[2]:


    commonPath = r"C:\Users\Sparsh Jain\Desktop\Capstone\final_app\farm\static\dataset\\"


    # In[3]:


    dataset = pd.read_csv(commonPath + "FinalDataset.csv")


    # In[4]:


    dataset_x = dataset.iloc[:,:-1]
    dataset_y = dataset.iloc[:, -1]

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y).reshape(-1,1)


    # In[5]:


    dataset_x.shape, dataset_y.shape


    # In[6]:


    dataset_x


    # In[7]:


    dataset_y


    # In[8]:


    onehotencoder = OneHotEncoder()
    dataset_y = onehotencoder.fit_transform(dataset_y).toarray()


    # In[9]:


    dataset_y


    # In[10]:


    x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size = 0.2, random_state = 100, shuffle = True)


    # In[11]:


    x_train.shape, x_test.shape, y_train.shape, y_test.shape


    # In[12]:


    type(x_train), type(x_test), type(y_train), type(y_test)


    # In[13]:


    x_train


    # In[14]:


    y_train


    # In[15]:


    x_test


    # In[16]:


    y_test


    # In[17]:


    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)


    # In[18]:


    n_input = 5;
    n_hidden_1 = 32
    n_hidden_2 = 64
    n_out = 5

    weights = {
        "h1" : tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=100)),
        "h2" : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], seed=100)),
        "out" :  tf.Variable(tf.random_normal([n_hidden_2, n_out], seed=100))
    }

    biases = {
        "h1" : tf.Variable(tf.random_normal([n_hidden_1], seed=100)),
        "h2" : tf.Variable(tf.random_normal([n_hidden_2], seed=100)),
        "out" :  tf.Variable(tf.random_normal([n_out], seed=100))
    }


    # In[19]:


    def forward_propagation(x_train, weights, biases):
        in_layer1 = tf.add(tf.matmul(x_train, weights['h1']), biases['h1'])
        out_layer1 = tf.nn.relu(in_layer1)
        
        in_layer2 = tf.add(tf.matmul(out_layer1, weights['h2']), biases['h2'])
        out_layer2 = tf.nn.relu(in_layer2)
        
        output = tf.add(tf.matmul(out_layer2, weights['out']), biases['out'])
        return output    


    # In[20]:


    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder(tf.int32, [None, n_out])
    pred = forward_propagation(x, weights, biases)


    # In[21]:


    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))


    # In[22]:


    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    optimize = optimizer.minimize(cost)


    # In[23]:


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    # In[24]:


    num_itr = 10000

    for i in range(num_itr):
        c, _ = sess.run([cost, optimize], feed_dict = {x : x_train, y : y_train})
        print(i, " : ", c)
        
        if c <0.5:
            break
            


    # In[25]:


    saver = tf.train.Saver()
    path_saved = saver.save(sess, r"C:\Users\Sparsh Jain\Desktop\Temp\weights.ckpt")


    # In[26]:


    path_saved


    # In[27]:


    x_test = scaler.transform(x_test)


    # In[34]:


    predictions = tf.argmax(pred, 1)
    correct_labels = tf.argmax(y, 1)
    correct_predictions = tf.equal(predictions, correct_labels)
    predictions_eval, correct_predictions = sess.run([predictions, correct_predictions], 
                                                      feed_dict = {x : x_test, y : y_test})
    return predictions_eval

@login_required
def CropRecommender(request):
    if request.method == 'POST':
        form = SoilForm(request.POST)
        if form.is_valid():
            Ph = form.cleaned_data.get('Ph')
            Nitrogen = form.cleaned_data.get('Nitrogen')
            Phosphorus = form.cleaned_data.get('Phosphorus')
            Pottasium = form.cleaned_data.get('Pottasium')
            Temprature = form.cleaned_data.get('Temprature')

            ClassNames = ["Maize","Sugarcane","Barley","Rice","Wheat","None"]

            LoadModelNB = joblib.load('D:\\Projects\\Captone 13 oct\\FarmAlert\\services\\NBCapstone.pkl')
            LoadModelDecisonTree = joblib.load('D:\\Projects\\Captone 13 oct\\FarmAlert\\services\\DTCapstone.pkl')
            LoadModelKNN = joblib.load('D:\\Projects\\Captone 13 oct\\FarmAlert\\services\\KNNCapstone.pkl')

            Input = [[Temprature,Ph,Nitrogen,Phosphorus,Pottasium],]

            PredNB = LoadModelNB.predict(Input)
            PredDT = LoadModelDecisonTree.predict(Input)
            PredKNN = LoadModelKNN.predict(Input)
            PredNN = predNN(Input)
            # Write Majority voting code

            MajorityVote = {0:0,1:0,2:0,3:0,4:0}

            print(PredNB[0],PredDT[0],PredKNN[0],PredNN[0])
            
            MajorityVote[PredNB[0]] += 1
            MajorityVote[PredDT[0]] += 1
            MajorityVote[PredKNN[0]] += 1
            MajorityVote[PredNN[0]] += 1

            maxCountIndex = 0
            maxCount = 0

            for crop,count in MajorityVote.items():
                print(crop,count)
                if count >= maxCount:
                    maxCount = count
                    maxCountIndex = crop


            if maxCount < 3:
                maxCountIndex = 5



            Context = {"Result" : ClassNames[maxCountIndex],
            }

            return render(request,'services/recommendation.html',Context)

    form = SoilForm()
    return render(request, 'services/crop.html', {'form':form})

@login_required
def Recommendation(request):
    return render(request,'services/recommendation.html')

def webScaper(govt_alert_url):

    # make connection, grab the page
    uclient = ureq(govt_alert_url)
    pg_html = uclient.read()
    uclient.close()

    # html parsing
    pg_soup = soup(pg_html, "html.parser")

    # grab all sections
    containers = pg_soup.findAll("div", {"class": "col-sm-4 col-md-4 col-lg-4"})
    # 1st section
    #container = containers[0]

    final_list = []
    table_row = containers[0].findAll("tr")
    rows = len(table_row)
    for i in range(0, rows - 1, 2):

        heading = table_row[i].a.text.strip()
        link = table_row[i].a["href"]
        t1 = ""
        text = table_row[i + 1].p.text.strip()
        for word in text.split():
            t1 = t1 + word + " "
        final_list.append((heading, link, t1))

    return final_list


@login_required
def govt_alert(request):

    final_list = webScaper("http://agriculture.gov.in/")
    context = {
        "list": final_list
    }

    return render(request, 'services/govtAlerts.html', context)


def getLocation():
    g = geocoder.ip('me')
    return g.latlng


@login_required
def cold_storages(request):

    API_KEY = "AIzaSyAIQUn1veJPYzWb-wAwu33-xOvCd05ZXHM"
    latlng = getLocation()

    lat1 = float(latlng[0])
    lng1 = float(latlng[1])

    google_places = GooglePlaces(API_KEY)

    gmaps = googlemaps.Client(key=API_KEY)
    google_places = GooglePlaces(API_KEY)
    query_result = gmaps.places_nearby(
        location='30.354907299999997,76.3677192',
        keyword='Coldstorage',
        radius=150000,
    )

    place_info = []
    lat_i = []
    lng_i = []
    name_array = []
    for place in query_result['results']:
        place_info.append((place['name'], place['rating']))
        lat_i.append((place['geometry']['location']['lat']))
        lng_i.append((place['geometry']['location']['lng']))
        name_array.append(place['name'])

    # lat_info = simplejson.dumps(lat_i)
    # lng_info = simplejson.dumps(lng_i)

    context = {
        "API_KEY": API_KEY,
        "place_info": place_info,
        "lat_info": lat_i,
        "lng_info": lng_i,
        "names": name_array,
    }

    return render(request, 'services/coldStorage.html', context)




def createDataset(commonPath):

    # reading the csv files of all the crops
    maize = pd.read_csv(commonPath + "Maize.csv", delimiter=",")
    sugarcane = pd.read_csv(commonPath + "Sugarcane.csv", delimiter=",")
    barley = pd.read_csv(commonPath + "Barley.csv", delimiter=",")
    rice = pd.read_csv(commonPath + "Rice.csv", delimiter=",")
    wheat = pd.read_csv(commonPath + "Wheat.csv", delimiter=",")

    # adding a label to every datapoint
    maize["Target"] = 0
    sugarcane["Target"] = 1
    barley["Target"] = 2
    rice["Target"] = 3
    wheat["Target"] = 4

    # concatenating all the crops to form a single dataset
    finalDataset = pd.concat([maize, sugarcane, barley, rice, wheat], axis=0)

    # creating the final dataset as a csv file to the "commonPath" specified
    finalDataset.to_csv(commonPath + "FinalDataset.csv", index=False)

def predNN(Input):
    LoadModelNB1 = joblib.load('D:\\Projects\\Captone 13 oct\\FarmAlert\\services\\NBCapstone.pkl')
    return LoadModelNB1.predict(Input)