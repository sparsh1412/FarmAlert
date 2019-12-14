from django.shortcuts import render, redirect
from django.contrib import messages
from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup
from django.contrib.auth.decorators import login_required
import geocoder
from googleplaces import GooglePlaces, types, lang
import googlemaps
import json as simplejson
from services.forms import SoilForm,YieldForm
import pandas as pd
from sklearn.externals import joblib 
from services.GetCropDataPoint import getCropDataPoint
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
import keras
import pandas as pd
import numpy as np
import sklearn
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

# Create your views here.


def build_regressor():
        regressor = Sequential()
        regressor.add(Dense(units=256, input_dim=216))
        regressor.add(Dense(units=256))
        regressor.add(Dense(units=128))
        regressor.add(Dense(units=1))
        adm = keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
        regressor.compile(optimizer=adm, loss='mean_squared_error',  metrics=['mse','accuracy'])
        return regressor

@login_required
def serve(request):
    return render(request, 'services/farmServices.html')


@login_required
def yieldPred(request):
    if request.method == 'POST':
        form = YieldForm(request.POST)
        if form.is_valid():
            Crop = form.cleaned_data.get('Crop')
            Location = form.cleaned_data.get('Location')
            crop_detail = getCropDataPoint(Location,Crop)
            Input = crop_detail
            pred = 0
            model2 = KerasRegressor(build_fn=build_regressor, epochs=10, batch_size=10, verbose=1)

            if Crop == "Barley":
                model2.model = load_model('D:\\Projects\\Clone 16 Nov Capstone\\FarmAlert\\farm\\static\\Kerasmodels\\Barleymodelkeras.h5')
            elif Crop == "Wheat":
                model2.model = load_model('D:\\Projects\\Clone 16 Nov Capstone\\FarmAlert\\farm\\static\\Kerasmodels\\Wheatmodelkeras.h5')
            elif Crop == "Maize":
                model2.model = load_model('D:\\Projects\\Clone 16 Nov Capstone\\FarmAlert\\farm\\static\\Kerasmodels\\Maizemodelkeras.h5')
            elif Crop == "Rice":
                model2.model = load_model('D:\\Projects\\Clone 16 Nov Capstone\\FarmAlert\\farm\\static\\Kerasmodels\\Ricemodelkeras.h5')
            else :
                model2.model = load_model('D:\\Projects\\Clone 16 Nov Capstone\\FarmAlert\\farm\\static\\Kerasmodels\\Sugarcanemodelkeras.h5')
            
            pred = model2.predict(Input)
            pred = pred[0]
        
            return render(request,'services/yieldResult.html',{'data':pred})

    form = YieldForm()
    return render(request, 'services/yieldPred.html', {'form':form})




@login_required
def CropRecommender(request):
    if request.method == 'POST':
        form = SoilForm(request.POST)
        if form.is_valid():
            Ph = form.cleaned_data.get('Ph')
            Nitrogen = form.cleaned_data.get('Nitrogen')
            Phosphorus = form.cleaned_data.get('Phosphorus')
            Potassium = form.cleaned_data.get('Potassium')
            Temperature = form.cleaned_data.get('Temperature')

            ClassNames = ["Maize","Sugarcane","Barley","Rice","Wheat","None"]

            LoadModelNB = joblib.load('D:\\Projects\\Captone 13 oct\\FarmAlert\\services\\NBCapstone.pkl')
            LoadModelDecisonTree = joblib.load('D:\\Projects\\Captone 13 oct\\FarmAlert\\services\\DTCapstone.pkl')
            LoadModelKNN = joblib.load('D:\\Projects\\Captone 13 oct\\FarmAlert\\services\\KNNCapstone.pkl')

            Input = [[Temperature,Ph,Nitrogen,Phosphorus,Potassium],]

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




