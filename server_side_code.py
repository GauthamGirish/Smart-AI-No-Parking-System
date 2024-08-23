import streamlit as st
from PIL import Image
import easyocr 
from ultralytics import YOLO
import time
import cv2
import requests
from io import BytesIO
from azure.storage.blob import BlobServiceClient as bsc
from pymongo import MongoClient as mc
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import config

#azure blob storage initialization
connect_str = config.AZURE_CONTAINER_KEY
container_name =config.AZURE_CONTAINER_NAME
blob_s_c = bsc.from_connection_string(conn_str=connect_str)
container_client = blob_s_c.get_container_client(container=container_name)

#Mongodb config
md_client = mc(config.MONGO_DATABASE_CONNECTION_STRING)
db=md_client.raspimg

model_weights = 'best.pt'
model = YOLO(model_weights, task='predict')
reader = easyocr.Reader(['en'])



# Function to send email
def send_email(plate_no=None):
    
    # Email credentials
    sender_email = config.EMAIL_ID
    sender_password = config.EMAIL_APPLICATION_PSWD

    # Recipient email address
    receiver_email = config.EMAIL_ID2

    # Create message container - the correct MIME type is multipart/alternative
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "License Plate Number Detected"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    # Create the plain-text and HTML version of the message
    text = f"License Plate Number Detected: {plate_no}"
    html = f"""\
    <html>
      <body>
        <p>License Plate Number Detected: <b>{plate_no}</b></p>
      </body>
    </html>
    """

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    msg.attach(part1)
    msg.attach(part2)

    # Send the message via SMTP server
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())


import numpy as np

def anpr(frame):
    # Convert the JpegImageFile object to a NumPy array
    frame_array = np.array(frame)
    
    results = model(frame_array)
    s = ''
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            roi = frame_array[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            text = reader.readtext(roi)
            for i in text:
                s=s+i[1]
                s = s.replace(" ", "")
            s=s.upper()
            # put box in cam
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # White color in BGR format
            thickness = 2
            
            # Write the text on the frame
            cv2.putText(frame_array, s, (x1, y1), font, font_scale, font_color, thickness)

            cv2.rectangle(frame_array, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
    # Convert the NumPy array back to an image
    result_image = Image.fromarray(frame_array)
    
    return result_image, s



if __name__=="__main__":
    #get url from mongo
    img_data = db.images.find_one({'img_id':1})
    img_url = img_data['url']

    #get image from blob
    response = requests.get(img_url)
    image = Image.open(BytesIO(response.content))

    #call prediction function
    image,plate_no=anpr(image)

    st.title(plate_no)
    st.image(image, caption='Captured Image', use_column_width=True)
    if(plate_no!=''):
        send_email(plate_no)

    time.sleep(5)
    st.experimental_rerun()