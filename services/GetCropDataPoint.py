import pandas as pd
import numpy as np

district = "patiala"
crop = "barley"

crop_season_dict = {
    "Barley": "Rabi",
    "Maize": "Kharif",
    "Rice": "Kharif",
    "Sugarcane": "AllYear",
    "Wheat": "Rabi",
}

locations = ["AMRITSAR",
             "BARNALA",
             "FARIDKOT",
             "FAZILKA",
             "FIROZEPUR",
             "GURDASPUR",
             "HOSHIARPUR",
             "JALANDHAR",
             "KAPURTHALA",
             "LUDHIANA",
             "MANSA",
             "MOGA",
             "MUKTSAR",
             "NAWANSHAHR",
             "PATHANKOT",
             "PATIALA",
             "RUPNAGAR",
             "SANGRUR"]

keep_cols = ["sunHour",
             "DewPointC",
             "cloudcover",
             "humidity",
             "precipMM",
             "pressure",
             "tempC",
             "visibility",
             "windspeedKmph"
             ]

kharif_lower_index = 20
kharif_upper_index = 36

sugarcane_lower_index = 0
sugarcane_upper_index = 51

rabi_lower_index_1 = 0
rabi_upper_index_1 = 8

rabi_lower_index_2 = 39
rabi_upper_index_2 = 51

temp = pd.get_dummies(locations)

one_hot_encoded_locations = {}

for i in range(len(locations)):
    one_hot_encoded_locations[locations[i]] = list(temp[locations[i]])

one_hot_encoded_locations


def getCropDataPoint(district, crop):
    district = district.upper()
    crop = crop.title()
    year = "2018-19"
    weather_path = r"D:\\Projects\\Clone 16 Nov Capstone\\FarmAlert\\farm\\static\\dataset\\Past Weather Data\\Week Wise Segregated"
    path = weather_path + "/" + year + "/" + district + ".csv"

    crop_data = []
    crop_data.extend(one_hot_encoded_locations[district])

    season = crop_season_dict[crop]

    data = pd.read_csv(path)

    if season == "Rabi":

        for col in keep_cols:
            crop_data.extend(list(data.iloc[rabi_lower_index_1:rabi_upper_index_1 + 1][col]))
            crop_data.extend(list(data.iloc[rabi_lower_index_2:rabi_upper_index_2 + 1][col]))

    elif season == "Kharif":

        for col in keep_cols:
            crop_data.extend(list(data.iloc[kharif_lower_index:kharif_upper_index + 1][col]))

    else:

        for col in keep_cols:
            crop_data.extend(list(data.iloc[sugarcane_lower_index:sugarcane_upper_index + 1][col]))

    crop_data = np.array(crop_data).reshape(1,-1)

    return crop_data


