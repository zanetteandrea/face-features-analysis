# Features extraction from faces - by Andrea Zanette
import cv2
import numpy as np
import argparse
import os
from emotion_net.emotionnet import EmotionDetectionNet
from ethnicity_net.ethnicitynet import EthnicityNet
import tensorflow as tf
import pandas as pd

tf.compat.v1.disable_eager_execution()

parser=argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to dataset")
parser.add_argument("-s", "--show", default=False, help="show predictions")
parser.add_argument("-c", "--csv", default=True, help="save results in a .csv file")
parser.add_argument("-n", "--name", default=True, help="the name of the dataset")
args=parser.parse_args()

# Scan dataset path
dataset_path = args.dataset

if not os.path.isdir(dataset_path):
    print("Invalid dataset path")
    exit()

n_elements = len([name for name in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, name))])
print("Find", n_elements, "elements")

ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['m','f']
ethnicitiesList = ['white', 'black', 'asian', 'indian', 'others']
emotionsList = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

weights_folder="./weights/"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
faceProto=weights_folder+"opencv_face_detector.pbtxt"
faceModel=weights_folder+"opencv_face_detector_uint8.pb"
ageProto=weights_folder+"age_deploy.prototxt"
ageModel=weights_folder+"age_net.caffemodel"
genderProto=weights_folder+"gender_deploy.prototxt"
genderModel=weights_folder+"gender_net.caffemodel"
ethnicityWeights=weights_folder+"ethnicity_detector_weights.h5"
emotionWeights=weights_folder+"emotion_detector_weights.h5"

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

ethnicityNet = EthnicityNet()
ethnicityNet.load_weights(ethnicityWeights)

emotionNet = EmotionDetectionNet()
emotionNet.load_weights(emotionWeights)

def detectFace(net, frame):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    
    index = np.argmax(detections[0, 0, :, 2])
    x1=int(detections[0,0,index,3]*frameWidth)
    y1=int(detections[0,0,index,4]*frameHeight)
    x2=int(detections[0,0,index,5]*frameWidth)
    y2=int(detections[0,0,index,6]*frameHeight)
    faceBox = [x1,y1,x2,y2]
    return faceBox

def detectGender(net, blobFace):
    net.setInput(blobFace)
    genderPreds = net.forward()
    index = genderPreds[0].argmax()
    return index
    
def detectAge(net, blobFace):
    net.setInput(blobFace)
    agePreds=net.forward()
    index = agePreds[0].argmax()
    return index

def detectEthnicity(model, face):
    im = cv2.resize(face, (200, 200))
    arr = np.reshape(im,(1,200,200,3))
    ethPred = model.predict(arr)
    index = np.argmax(ethPred)
    return index

def detectEmotion(model, face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resizedFace = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)

    pred = model.predict(resizedFace)
    index = int(pred.argmax(axis=1))

    return index


print("Start detecting faces")
padding = 20
images_iterator = os.scandir(dataset_path)
results = []
for img in images_iterator:

    try:
        image = cv2.imread(img.path)
        
        faceBox = detectFace(faceNet, image)

        face=image[max(0,faceBox[1]-padding):min(faceBox[3]+padding, image.shape[0]-1), max(0,faceBox[0]-padding):min(faceBox[2]+padding, image.shape[1]-1)]
        blobFace=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        
        gender = detectGender(genderNet, blobFace)
        age = detectGender(ageNet, blobFace)
        ethnicity = detectEthnicity(ethnicityNet, face)
        emotion = detectEmotion(emotionNet, face)

        if args.csv:
            results.append((img.name, gender, age, ethnicity, emotion))

        if args.show:
            cv2.rectangle(image, (faceBox[0],faceBox[1]), (faceBox[2],faceBox[3]), (0,255,0), 3, 8)
            cv2.putText(image, f'{genderList[gender]}, {ageList[age]}, {ethnicitiesList[ethnicity]}, {emotionsList[emotion]}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow(img.name, image)
            cv2.waitKey()
            cv2.destroyAllWindows()
    except Exception as err:
        print("Errore con", img.name, err)

if args.csv:
    df = pd.DataFrame(results, columns=["name", "gender", "age", "ethnicity", "emotion"])
    df.to_csv("./results/features_"+args.name+"_dataset_results.csv", index=False)
