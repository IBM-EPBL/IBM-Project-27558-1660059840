from flask import Flask,render_template,request
import cv2
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import load_model
import numpy as np
app = Flask(__name__,template_folder="Templates")

from ibm_watson_machine_learning import APIClient
wml_credentials={
    "url":'https://us-south.ml.cloud.ibm.com',
    "apikey":'wrWJKcS_-bKdJ7LajFGsaxu9Sk2nCDDuMZcdUUP6p2iD'
}
client=APIClient(wml_credentials)

def guid_from_space_name(client,space_name):
    space=client.spaces.get_details()
    return(next(item for item in space['resources'] if item['entity']['name']==space_name)['metadata']['id'])
                              
                              
space_uid=guid_from_space_name(client,'disaster')
client.set.default_space(space_uid)

client.repository.download("c3820a6b-fd74-48cd-8b0c-70d4ffe1c440","disater.taz.gz")

model=load_model("ibm_img_model.h5.h5")

#print(model)
def ran(result):
  if(result=='Cyclone'):
      return 2
  elif (result=='Earthquake'):
      return 3
  elif (result=='Flood'):
      return 1
  elif (result=='Wildfire'):
      return 4
@app.route('/',methods=['GET'])
def index():
  return render_template('home.html')

@app.route('/home',methods=['GET'])
def home():
  return render_template('home.html')

@app.route('/intro',methods=['GET'])
def about():
  return render_template('intro.html')

@app.route('/run',methods=['GET'])
def upload():
  return render_template('run.html')

@app.route('/uploader',methods=['GET','POST'])
def predict():
  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture(0)
  
  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
  # start = time.process_time() 
  # Read until video is completed
  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      frame=cv2.flip(frame,1)
      output = frame.copy()
      frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      frame = cv2.resize(frame,(64,64))
      x=np.expand_dims(frame,axis=0)
      result = np.argmax(model.predict(x),axis=1)
      index=['Cyclone','Earthquake','Flood','Wildfire']
      result = str(index[result[0]])
      #print(result)
      res=ran(result)
      cv2.putText(output,"Intensity: {}".format(res),(10,120),cv2.FONT_HERSHEY_PLAIN,1,(0,25,255),1)
      cv2.putText(output,"Disaster: {}".format(result),(10,100),cv2.FONT_HERSHEY_PLAIN,1,(0,25,255),1)
      # Display the resulting frame
      cv2.imshow('Frame',output) 
      # Press Q on keyboard to  exit
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
    # Break the loop
    else: 
      break
  # When everything done, release the video capture object
  cap.release()
  
  # Closes all the frames
  cv2.destroyAllWindows()
  return render_template('run.html')

if __name__ == '__main__':
  app.run(host='0.0.0.0',port=8000,debug=False)