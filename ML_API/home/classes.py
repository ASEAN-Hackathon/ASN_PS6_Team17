import tensorflow as tf
from tensorflow.keras.models import load_model,Sequential,Model
import cv2
import pandas as pd
from pprint import pprint
import os
import numpy as np
import pickle
import sklearn
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
import time

def embedding(i,model):#embedding of an image (the input is an image and the model)
    print("The image given as input is {}".format(i))
    image = cv2.imread(i)
    print(image.shape)
    img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = img.reshape(1, 256, 256, 3)
    embedding = model.predict(image)
    return embedding


def category(df,cat_arr,cat_name,model):#embedding of all the images in a category with different classes
                                        #(the input is the dataframe, arr of classes in category, category name, model) 

    cat_arr = [x for x in cat_arr if str(x) != 'nan']
    pprint(cat_name)
    if str(cat_name) == 'family':
        cat_number = 2  
    elif str(cat_name) == 'genus':
        cat_number = 3
    elif str(cat_name) == 'species':
        cat_number = 4    
    
    print(str(cat_name))
    print("The catyegory number is ",cat_number)
    for ele in cat_arr:
        family_category = []
        for i in range(len(df)):
            print("The name of the element in dataframe is {} and the element is {} ".format(df.iloc[i,cat_number],ele))
            if str(df.iloc[i,cat_number]) == str(ele):
                print("In the loop")
                img_name = str(df.iloc[i,1])
                print("The name of the image is {}".format(img_name))
                e = embedding(os.path.join('Dataset/FDFML/crops/',str(img_name)),model)                        
                print("The dimension of the embedding is {}".format(e.shape))
                family_category.append(e)
        print("The family category shape is {}".format(np.array(family_category).shape))
        np.savez('results/'+cat_name+'/'+ele+'.npz',family_category)     


def mean_category():

    for cat in os.listdir('results'):
        dic = {}
        if cat == 'epoch_19550.h5':
                continue
        else:        
            for class_ in os.listdir('results/'+str(cat)):            
                data = np.load('results/'+cat+'/'+class_,allow_pickle = True)
                a = data.f.arr_0
                print(type(np.array(a)))
                print(a.shape)
                dic[str(class_)]= np.mean(a,axis = 0)
            pickle.dump(dic,open('results/'+cat+'/'+'mean.pickle','wb'))    
        

def evaluation(df,cat_name):
    data_mean = pickle.load(open('results/'+cat_name+'/'+'mean.pickle','rb'))
    print("The type of the file is {} and the length is {}".format(type(data_mean),data_mean.items()))
    true,false = 0,0
    
    #accessing the embedded images 
    for file_name in os.listdir('results/'+cat_name+'/'):
        if ".npz" in str(file_name):  
            a = np.load('results/'+cat_name+'/'+file_name,allow_pickle = True)
            data = a.f.arr_0
            #print(data.shape[0])
            for i in range(data.shape[0]): #accessing the image in data_arr         
                dic_predicted = {}
                '''for k,v in data_mean.items():
                    print("The dimension of the v is {} and of data is {}".format(np.array(v[0,0,0]).shape,np.array(data[i,0,0,0]).shape))
                    class_name = str(k).split('.npz')[0]
                    distance = cosine_similarity(v.reshape([512,1]),data.reshape([512,-1]))
                    print(distance) 
                    dic_predicted[class_name] = distance'''
                key = list(data_mean.keys())
                values = np.array(list(data_mean.values()))
                #pprint(values)
                #pprint(key)
                #print("The dimension of the v is {} and of data is {}".format(values.shape,np.array(data[i,0,0,0]).shape))    
                '''distance = cosine_similarity(data[i,0,0,0].reshape([1,512]),values.reshape([-1,512]))'''
                distance = euclidean_distances(data[i,0,0,0].reshape([1,512]),values.reshape([-1,512]))    
                pred_class = key[np.argmin(distance)]

                #pprint(dic_predicted)
                '''values,keys = zip(*sorted(zip(dic_predicted.values(),dic_predicted.keys())))
                pprint("The class name from the file processed is {}".format(class_name))'''
                #print("The true class is  {} and predicted is {}".format(pred_class,str(file_name).split(".npz")[0]))
                #print(str(pred_class).split(".npz")[0],str(file_name).split(".npz")[0],sep = "                                     ")
                if str(pred_class).split(".npz")[0] == str(file_name).split(".npz")[0]:
                    true+=1
                else:
                    false+=1

                pprint("This is the value of true {} and false {}".format(true,false))             
                    
        else:
            continue

def preparation(directory_path):
    
    X,Y = [],[]
    for class_ in os.listdir(directory_path):
        if ".npz" in str(class_): 
            x = np.load(os.path.join(directory_path,class_),allow_pickle = True)
            x_train = x.f.arr_0 
            number_of_images = x_train.shape[0]
            X.extend(list(x_train))
            Y.extend([str(class_).split(".npz")[0]]*number_of_images)
            #print("The shape of x and y will be {} and {}".format(np.array(X).shape,np.array(Y).shape))
        else:
            continue
    return np.array(X).reshape([-1,512]),np.array(Y)
