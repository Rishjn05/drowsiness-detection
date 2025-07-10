#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[3]:


pip install dlib-19.24.1-cp311-cp311-win_amd64.whl


# In[1]:





# In[3]:





# In[2]:


import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import face_recognition
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')


# In[13]:


img_path="test2.jpeg"
img=Image.open(img_path)
plt.axis('off')
plt.imshow(img)
plt.show()


# In[14]:


img_path2="test1"
img2=Image.open(img_path2)
plt.axis('off')
plt.imshow(img2)
plt.show()


# In[15]:


def highlight_facial_points(img_path):
    img_bgr=cv2.imread(img_path)
    img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    face_locations=face_recognition.face_locations(img_rgb,model='cnn')
    for face_loc in face_locations:
        landmarks=face_recognition.face_landmarks(img_rgb,[face_loc])[0]
        for landmark_type,landmark_points in landmarks.items():
            for (x,y) in landmark_points:
                cv2.circle(img_rgb,(x,y),3,(0,255,0),-1)
                
    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
    


# In[16]:


highlight_facial_points(img_path)


# In[17]:


highlight_facial_points(img_path2)


# In[18]:


def eye_aspect_ratio(eye):
    A=distance.euclidean(eye[1],eye[5])
    B=distance.euclidean(eye[2],eye[4])
    C=distance.euclidean(eye[0],eye[3])
    ear=(A+B)/(2.0*C)
    return ear

def mouth_aspect_ratio(mouth):
    A=distance.euclidean(mouth[2],mouth[8])
    B=distance.euclidean(mouth[1],mouth[4])
    C=distance.euclidean(mouth[0],mouth[5])
    mar=(A+B)/(2.0*C)
    return mar


# In[19]:


def process_image(frame):
    ear_threshold=0.2
    mouth_threshold=0.6
    if frame is None:
        raise ValueError('Image is not found or unable to open.')
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations=face_recognition.face_locations(rgb_frame)
    
    eye_flag=mouth_flag=False
    
    for face_location in face_locations:
        landmarks=face_recognition.face_landmarks(rgb_frame,[face_location])[0]
        left_eye=np.array(landmarks['left_eye'])
        right_eye=np.array(landmarks['right_eye'])
        mouth=np.array(landmarks['bottom_lip'])
        
        left_ear=eye_aspect_ratio(left_eye)
        right_ear=eye_aspect_ratio(right_eye)
        ear=(left_ear+right_ear)/2.0
        mar=mouth_aspect_ratio(mouth)
        
        if ear<ear_threshold:
            eye_flag=True
            
        if mar>mouth_threshold:
            mouth_flag=True
        
    return eye_flag,mouth_flag


# In[20]:


img2=cv2.imread(img_path2)
process_image(img2)


# In[21]:





# In[22]:


from playsound import playsound
import threading


# In[23]:


def play_alarm():
    playsound('An aggressive car alarm featuring rapid, high-decibel beeps followed by a low, echoing tone..wav')  # alarm sound file in same folder


# In[24]:


import time


# In[26]:


video_path='istockphoto-1217195460-640_adpp_is.mp4'
#video_cap=cv2.VideoCapture(0)  #for getting video from webcam
video_cap=cv2.VideoCapture(video_path)
count=score=0
start_time = time.time()

drowsy_frames = 0
while True:
    success,img=video_cap.read()
    if not success:
        break
        
    img=cv2.resize(img,(800,500))
    
    count+=1
    n=5
    if count%n==0:
        eye_flag,mouth_flag=process_image(img)
        if eye_flag or mouth_flag:
            score+=1
            drowsy_frames+=1;
        else:
            score-=1
            if score<0:
                score=0
    font=cv2.FONT_HERSHEY_SIMPLEX
    text_x=10
    text_y=img.shape[0]-10
    text=f"Score:{score}"
    cv2.putText(img,text,(text_x,text_y),font,1,(0,0,255),2,cv2.LINE_AA)
    
    if score>=5:
        text_x=img.shape[1]-130
        text_y=40
        text="Drowsy"
        cv2.putText(img,text,(text_x,text_y),font,1,(0,0,255),2,cv2.LINE_AA)
        threading.Thread(target=play_alarm).start()
        print(f"Alarm rings")
        
        
    cv2.imshow('Drowsiness detection',img)
    if cv2.waitKey(1) & 0xFF!=255:
        break
        
end_time = time.time()
duration = end_time - start_time
print(f"Processed {count} frames in {duration:.2f} seconds")
print(f"Drowsy frames: {drowsy_frames} ({(drowsy_frames/count)*100:.2f}%)")
        
video_cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




