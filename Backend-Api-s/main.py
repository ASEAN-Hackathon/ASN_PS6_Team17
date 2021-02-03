from flask import Flask,jsonify,request
import firebase_admin
from firebase_admin import credentials,firestore,storage
import cv2
import base64
import requests
import matplotlib.pyplot as plt
import io
import numpy as np
import datetime  
from haversine import Unit
import haversine as hs
import random
from PIL import Image
import pyrebase
import os

# To be changed.
BASE_PATH = "/home/learner/Desktop/app2/backend/images/"

def convert(string,name):

    image = base64.b64decode(str(string))       
    fileName = name

    imagePath = BASE_PATH + fileName

    img = Image.open(io.BytesIO(image))
    img.save(imagePath, 'jpeg')
    return fileName


app = Flask(__name__)

cred = credentials.Certificate('serviceAccount.json')

firebase_admin.initialize_app(cred)

config = {
    "apiKey": "AIzaSyDCHQLRGASmP2PhgQY0RAeCu2kn6B8c21Y",
    "authDomain": "fishapp-122bc.firebaseapp.com",
    "projectId": "fishapp-122bc",
    "databaseURL": "https://fishapp-122bc-default-rtdb.firebaseio.com/",
    "storageBucket": "fishapp-122bc.appspot.com",
    "messagingSenderId": "997413319923",
    "appId": "1:997413319923:web:3397d9ddbd9f4e28db3386",
    "measurementId": "G-CRGEX1F1X9"
}


firebase = pyrebase.initialize_app(config)

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

db = firestore.client()

# Document for userAccounts
userAccounts  = db.collection('userAccounts')
userCatches = db.collection('userCatches')
adminUpdates = db.collection('adminUpdates')
fishData = db.collection('fishData')

#The base route - This usually is used to check whether server is up or not.
@app.route('/')
def index():
    return "Hello World"


@app.route('/signup',methods=['POST'])
def signup():

    data = request.json
    print(data)
    try:
        getMobileNo = str(data['mobileNo'])
        # We get the user
        user = userAccounts.document(getMobileNo).get().to_dict()
        if user != None:
            return {"success":False,"details":"Mobile Number Already Exists"}
        else:
            # Creating the account details in firebase
            userAccounts.document(getMobileNo).set(data)
            userCatches.document(getMobileNo).set(None)
            response = jsonify({"success":True})
            return response
    except Exception as e:
        return f"An Error Occured: {e}",400


'''

    Month v/s number of catches on a Histogram.
    Month - Number, we are giving.
    Catch quantity - Number

    Not Optmized - Bad DB Structure.

'''
@app.route('/monthVsCatch',methods=['GET'])
def getChart1():
    try:
        data = fishData.stream()
        dataDict = {}
        for data1 in data:
            data1 = data1.to_dict()
            month = int(data1['date'].split("-")[1])
            count = 0
            for i in data1['catches']:
                count += int(i['quantity'])

            if month in dataDict:
                dataDict[month] = dataDict[month] + count
            else:
                dataDict[month] = count
        plt.bar(list(dataDict.keys()), dataDict.values(), color='g')
        plt.xlabel('Months')
        plt.ylabel('No of Catches')
        plt.title('Month Vs No of catches caught')
        a = 'temp'+ ''.join([str(random.randint(0, 999)).zfill(3) for _ in range(2)]) +'.png'
        imageTempPath = BASE_PATH + a
        plt.savefig(imageTempPath)
        storage.child('charts/{}'.format(a)).put(imageTempPath)
        chartUrl = storage.child('charts/{}'.format(a)).get_url(None)
        print(chartUrl)
        return jsonify({'chartUrl':chartUrl,'average':res,'moneySaved':res*2})

    except Exception as e:
        return f"An Error Occured", 500

'''

    Month v/s weight of catches on a histogram
    Month - Number, we are giving.
    Weight - In Kgs

    Not Optmized - Bad DB Structure.
'''
@app.route('/monthVsWeight',methods=['GET'])
def getChart2():

    try:
        data = fishData.stream()
        dataDict = {}
        for data1 in data:
            data1 = data1.to_dict()
            month = int(data1['date'].split("-")[1])
            weight = data1['totalWeight']
            if month in dataDict:
                dataDict[month] = dataDict[month] + weight
            else:
                dataDict[month] = weight
        plt.bar(list(dataDict.keys()), dataDict.values(), color='g')
        plt.xlabel('Month')
        plt.ylabel('Weight of catch caught')
        a = 'temp'+ ''.join([str(random.randint(0, 999)).zfill(3) for _ in range(2)]) +'.png'
        imageTempPath = BASE_PATH + a
        plt.title('Month vs Weight of catch caught')
        plt.savefig(imageTempPath)
        storage.child('charts/{}'.format(a)).put(imageTempPath)
        chartUrl = storage.child('charts/{}'.format(a)).get_url(None)
        print(chartUrl)
        return jsonify({'chartUrl':chartUrl})
    except Exception as e:
        print(e)
        return f"An Error Occured", 500

'''

    Month v/s CPUE - ForAdmin

'''
@app.route('/monthVsCpue',methods=['GET'])
def getChart3():
    data = fishData.stream()
    dataDict = {}
    monthCountDict = {}
    try:
        for data1 in data:
            data1 = data1.to_dict()
            print(data1)
            month = int(data1['date'].split("-")[1])
            weight = data1['totalWeight']
            hours = data1['hours']
            CPUE = weight/hours
            if month in monthCountDict:
                monthCountDict[month] += 1
            else:
                monthCountDict[month] = 1

            if month in dataDict:
                dataDict[month] = dataDict[month] + CPUE
            else:
                dataDict[month] = CPUE
        ans = {}
        for i in dataDict.keys():
            ans[i] = dataDict[i]/monthCountDict[i]
        plt.bar(list(ans.keys()), ans.values(), color='g')
        plt.xlabel('Months')
        plt.ylabel('CPUE(weight of catch/hour)')
        a = 'temp'+ ''.join([str(random.randint(0, 999)).zfill(3) for _ in range(2)]) +'.png'
        imageTempPath = BASE_PATH + a
        plt.savefig(imageTempPath)
        plt.title("Month Vs CPUE")
        storage.child('charts/{}'.format(a)).put(imageTempPath)
        chartUrl = storage.child('charts/{}'.format(a)).get_url(None)
        print(chartUrl)
        return jsonify({'chartUrl':chartUrl,})
    except Exception as e:
        return f"An Error Occured", 500        

'''

    Month V/s CPUE - For Fisherman

'''
@app.route('/monthVsCpueForFisherman',methods=['GET'])
def getChart4():
    print(request.args)
    try:
        number = str(request.args['number'])
        catchDetails = userCatches.document(number).get().to_dict()["catches"]
        # print(catchDetails)
        resDict = dict()
        monthDict = dict()
        for i in catchDetails:

            hours = float(i['hours'])
            weight = float(i['weight'])
            month = int(i['date'].split("-")[1])
            print(month)
            CPUE = weight/hours
            print(CPUE)
            if month in resDict:
                resDict[month] = resDict[month] +  CPUE
            else:
                resDict[month] = CPUE
            if month in monthDict:
                monthDict[month] += 1

            else:
                monthDict[month] = 1


            print(resDict,monthDict)
        ans = {}
        total = 0
        for i in resDict.keys():
            ans[i] = resDict[i]/monthDict[i]
            total += ans[i]
        
        # This is the average weight/hour.
        # Money saved is random.
        res = total/len(ans)
        print(str(ans.keys())+" "+str(ans.values())+"hi")
        plt.bar(list(ans.keys()), ans.values(), color='g')
        plt.title("Month Vs CPUE - For Fisherman")
        plt.xlabel('Months')
        plt.ylabel('CPUE(weight of catch/hour)')
        a = 'temp'+ ''.join([str(random.randint(0, 999)).zfill(3) for _ in range(2)]) +'.png'
        imageTempPath = BASE_PATH + a
        plt.savefig(imageTempPath)
        storage.child('charts/{}'.format(a)).put(imageTempPath)
        chartUrl = storage.child('charts/{}'.format(a)).get_url(None)
        print(chartUrl)
        return jsonify({'chartUrl':chartUrl,'average':res,'moneySaved':res*2})
    except Exception as e:
        return f"An Error Occured", 500



@app.route('/postAnUpdate',methods=['POST'])
def postUpdate():

    data  = request.json
    print(data)
    try:
        current_time = datetime.datetime.now() 
        updateId = ''.join([str(random.randint(0, 999)).zfill(3) for _ in range(2)])
        date = '/'.join([str(current_time.day),str(current_time.month),str(current_time.year)])
        time = ':'.join([str(current_time.hour),str(current_time.minute),str(current_time.second)])
        data['updateTime'] = time
        data['updateDate'] = date
        data['updateId'] = updateId
        adminUpdates.document(updateId).set(data)
        response = jsonify({"success":True}),200
        return response
    except Exception as e:
        return f"An Error Occured", 500


@app.route('/getAllUpdates',methods=['GET'])
def getAllUpdates():
    try:
        updates = adminUpdates.stream()
        updatesArray = list()
        for update in updates:
            temp = update.to_dict()
            updatesArray.append(temp)
        updatesArray.reverse()
        print(updatesArray)
        return jsonify({'success':True,'updatesArray':updatesArray}), 200

    except Exception as e:
        return f"An error occured", 500

'''

    #How will a particular catch look like.
    {
        "image":"imageUrl",
        "date":date,
        "name":name,
        "weight":weight,
    }

'''

#We need to show the users catches.
@app.route('/getCatchesHistory',methods=['GET'])
def getCatches():
    print("hi")
    data = request.args
    print(data)
    try:
        mobile = str(data['mobile'])
        print(mobile)
        catchDetails = userCatches.document(mobile).get().to_dict()["catches"]
        print(catchDetails)
        catchDetails.reverse()
        return jsonify({"success":True,"catches":catchDetails})
    except Exception as e:
        return f"An error occured {e}", 400


@app.route('/getNearbyCatches',methods=['GET'])
def getNearbyCatches():
    print(request.args)
    latitude = float(request.args['latitude'])
    longitude = float(request.args['longitude'])
    nearbyCatches = fishData.stream()
    nearestCatches = list()
    for catch in nearbyCatches:
        # print(catch.to_dict(),"hi")
        temp = catch.to_dict()
        loc1 = (float(temp['latitude']),float(temp['longitude']))
        loc2 = (latitude,longitude)
        distance = hs.haversine(loc1,loc2,unit=Unit.METERS)
        print(distance)
        if distance < 25000:
            nearestCatches.append(temp)
            pass
        else:
            pass
    return jsonify({'success':True,'nearestCatches':nearestCatches})


@app.route('/updateCatch',methods=['POST'])
def updateCatch():

    try:
        data  = request.json
        print(data)
        print(data['image'])
        convert(data['image'],data['imageFileName'])
        imagePath = BASE_PATH + data['imageFileName']
        
        storage.child('fishes/{}'.format(data['imageFileName'])).put(imagePath)
        fish_url = storage.child('fishes/{}'.format(data['imageFileName'])).get_url(None)
        # The url of the image sent from app
        print(fish_url)
        request_url = 'http://35.240.219.8:8080/register?key='+fish_url
        print(request_url)
        try:
            predictedData = requests.get(request_url).json()
        except Exception as e:
            print(e)
        print(predictedData)
        temp = int(''.join([str(random.randint(0, 999)).zfill(3) for _ in range(2)]))
        data2 = {
            'date':data['date'],
            'description':data['description'],
            'image':fish_url,
            'hours':float(data['hours']),
            'latitude':float(data['latitude']),
            'longitude':float(data['longitude']),
            'name':data['name'] + " "+ str(temp),
            'weight':float(data['weight']),
            'number':data['number'],
            'catchId': temp,
            'catches':predictedData['fishData'],
            'boundedImageUrl':predictedData['boundedUrl']
        }
        ref = userCatches.document(str(data['number']))
        ref.update({u'catches': firestore.ArrayUnion([data2])})
        print(data2)
        data1 = {
            'latitude':float(data2['latitude']),
            'longitude':float(data2['longitude']),
            'totalWeight':float(data2['weight']),
            'date':data2['date'],
            'catchId':data2['catchId'],
            'hours':data2['hours'],
            'catches':predictedData['fishData'],
            'boundedImageUrl':predictedData['boundedUrl']
        }
        print(data1)
        fishData.document(str(data1['catchId'])).set(data1)
        return jsonify({"success":True})
    
    except Exception as e:
        print(e)
        return jsonify({"success":False}), 400


@app.route('/login',methods=['POST'])
def login():

    data = request.json
    print(data)

    try:
        getMobileNo = str(data['mobile'])
        user = userAccounts.document(getMobileNo).get().to_dict()
        print(user)
        if user == None:
            return {"success":False,"details":"Mobile Number Not Registered"}
        else:
            print("hi")
            passwordSent = data['password']
            actualPassword = user['password']
            print(passwordSent,actualPassword)
            if passwordSent == actualPassword:
                user['success'] = True
                response = jsonify(user)
                return response
            else:
                response = jsonify({"success":False,"details":"Invalid Password"})
                return response
    
    except Exception as e:
        return f"An error Occured: {e}",400

if __name__ == "__main__":
    app.run(debug=True)
