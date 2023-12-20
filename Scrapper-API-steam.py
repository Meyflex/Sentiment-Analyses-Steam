import csv
import requests
import json


def get_app_ID():
    return requests.get(url='http://api.steampowered.com/ISteamApps/GetAppList/v0002/', params={'json': 1},
                        headers={'User-Agent': 'Mozilla/5.0'}).json()
def get_reviews(appid, params=None):
    if params is None:
        params = {'json': 1}
    return requests.get(url='https://store.steampowered.com/appreviews/' + appid, params=params,
                        headers={'User-Agent': 'Mozilla/5.0'}).json()


def get_game_reviews(appid, n=100):
    reviews = []
    cursor = '*'
    params = {
        'json': 1,
        'filter': 'all',
        'language': 'english',
        'day_range': 9223372036854775807,
        'review_type': 'all',
        'purchase_type': 'all'
    }
    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(100, n)
        n -= 100

        response = get_reviews(appid, params)
        cursor = response['cursor']
        reviews += response['reviews']

        if len(response['reviews']) < 100: break

    return reviews


def get_app_ID():
    return requests.get(url='http://api.steampowered.com/ISteamApps/GetAppList/v0002/', params={'json': 1},
                        headers={'User-Agent': 'Mozilla/5.0'}).json()


apps = get_app_ID()
reviews = [['review', 'voted_up']]

for i in range(len(str(apps['applist']['apps']))):
    reviews += [[r['review'], r['voted_up']] for r in (get_game_reviews(str(apps['applist']['apps'][i]['appid'])))]
    if len(reviews) >= 100000:
        break

with open('Steam_Reviews.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(reviews)

print(len(reviews))
print(reviews)
