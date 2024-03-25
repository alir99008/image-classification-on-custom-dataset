
import numpy as np 
from matplotlib import pyplot as plt
import cv2
import os

import pandas as pd

import glob


size=128

train_images=[]
train_labels = []

for directory_path in glob.glob("training/*"):
    
    label = directory_path.split("\\")[-1] 
    print(label)
    for img_path in glob.glob(os.path.join(directory_path , "*.jfif")):
        print(img_path) 
        img = cv2.imread(img_path)      
        img = cv2.resize(img , (size , size))    
        train_images.append(img)   
        train_labels.append(label)  
        
train_images = np.array(train_images)     
train_labels = np.array(train_labels)




#test images 

test_images=[]
test_labels = []

for directory_path in glob.glob("testing/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    print(directory_path)
    for img_path in glob.glob(os.path.join(directory_path , "*.jfif")):
        print(img_path) 
        img = cv2.imread(img_path)
        img = cv2.resize(img , (size , size))
        test_images.append(img)
        test_labels.append(label)
        
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)






from sklearn import preprocessing
le = preprocessing.LabelEncoder()   
le.fit(train_labels)   
train_label_encoded = le.transform(train_labels)      #es step main hum ny 0 ,1,2,3,4 main convert kr diya

le.fit(test_labels)      
test_label_encoded = le.transform(test_labels)      




x_train , y_train , x_test , y_test = train_images , train_label_encoded , test_images , test_label_encoded



# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0    


def feature_extraction(dataset):      
    image_dataset = pd.DataFrame()
    
    for image in range(dataset.shape[0]):                 
        #print(img)
        df=pd.DataFrame()   
        
        input_img = dataset[image , : , : , :]       
        
        img = input_img  
        
        pixel_values = img.reshape(-1)      
        df["pixel value"] = pixel_values     
        
        
        #Gabor filter
        num=1
        for theta in range(2):
            theta = theta/4. * np.pi
            for sigma in (1,3):
                for lamda in np.arange(0 , np.pi , np.pi/4):
                    for gamma in (0.05 , 0.5):
                        
                        ksize = 9 
                        gabor_label = "gabor"+ str(num)
                        kernel = cv2.getGaborKernel((ksize , ksize),sigma , theta , lamda , gamma , 0 , ktype=cv2.CV_32F)
                        
                        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                        filtered_img = fimg.reshape(-1)
                        
                        df[gabor_label] = filtered_img
                        
                        num = num+1
                        
                        
        
       
        
        from scipy import ndimage as nd

        
        gaussian_img = nd.gaussian_filter(img, sigma=3)
        gaussian_img_reshape = gaussian_img.reshape(-1)

        df["gaussian sigma 3"] = gaussian_img_reshape


        
        gaussian_img2 = nd.gaussian_filter(img, sigma=7)
        gaussian_img2_reshape = gaussian_img2.reshape(-1)

        df["gaussian sigma 7"] = gaussian_img2_reshape


         
        
        median_img = nd.median_filter(img , size = 3)
        median_img_reshape = median_img.reshape(-1)
        df["median image"] = median_img_reshape
       
        image_dataset = image_dataset.append(df)          
        
    return image_dataset    






image_features = feature_extraction(x_train)     


image_features = np.expand_dims(image_features , axis=0)      
x_for_RF  = np.reshape(image_features,(x_train.shape[0] ,-1))




from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 30, random_state = 42)     

RF_model.fit(x_for_RF  , y_train)     




test_features = feature_extraction(x_test)           



test_features = np.expand_dims(test_features , axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1)) 



test_prediction = RF_model.predict(test_for_RF)
print("test prediction ",test_prediction)



test_prediction = le.inverse_transform(test_prediction)          


from sklearn import metrics
print("Accuracy ",metrics.accuracy_score(test_labels,test_prediction))



#Print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, test_prediction)

print("Confusion matrix:")
print(cm)


#Check results on a few random images
import random

n=random.randint(0, x_test.shape[0]-1) 
print("n = ",n)
img = x_test[n]     
plt.imshow(img)

#Extract features and reshape to right dimensions
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_features=feature_extraction(input_img)          
input_img_features = np.expand_dims(input_img_features, axis=0)     
input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))     
#Predict
img_prediction = RF_model.predict(input_img_for_RF)      
img_prediction = le.inverse_transform([img_prediction]) 
print("The prediction for this image is: ", img_prediction)
print("The actual label for this image is: ", test_labels[n])
