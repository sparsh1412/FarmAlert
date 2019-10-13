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

# Create your views here.


@login_required
def serve(request):
    return render(request, 'services/farmServices.html')


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

            print(Ph, Nitrogen, Phosphorus, Pottasium, Temprature)

    form = SoilForm()

    return render(request, 'services/crop.html', {'form': form})


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


commonPath = r"C:\Users\Sparsh Jain\Desktop\Capstone\final_app\farm\static\dataset\\"


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
