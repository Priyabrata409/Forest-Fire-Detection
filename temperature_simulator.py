import requests
import json

base_url = "http://api.openweathermap.org/data/2.5/weather?"
with open("api_key.json") as f:
    api_key = json.load(f)["key"]
city_name = "bhubaneswar"
complete_url = base_url + "appid=" + api_key + "&q=" + city_name


def get_weather_data():
    res = requests.get(complete_url)
    data = res.json()
    data = data["main"]
    print(data)
    temp = data["temp"]
    humidity = data["humidity"]
    pressure = data["pressure"]
    temp = temp - 273.15
    return temp, pressure, humidity
