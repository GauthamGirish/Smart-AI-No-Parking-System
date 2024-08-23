import cv2
import time
from azure.storage.blob import BlobServiceClient
from pymongo import MongoClient
import numpy as np
import io
import config

# Azure setup
connect_str = config.AZURE_CONTAINER_KEY
container_name =config.AZURE_CONTAINER_NAME
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)

# MongoDB setup
mongo_client = mc(config.MONGO_DATABASE_CONNECTION_STRING)
db = mongo_client.raspimg

# Initialize camera
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture image from webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to byte format for uploading
        _, buffer = cv2.imencode('.jpg', frame)
        byte_image = io.BytesIO(buffer)

        # Create a unique blob name
        blob_name = f'img_{int(time.time())}.jpg'

        # Upload image to Azure
        blob_client = container_client.upload_blob(blob_name, byte_image, overwrite=True)
        url = f"https://{connect_str.split(';')[1].split('=')[1]}.blob.core.windows.net/{container_name}/{blob_name}"

        # Save URL in MongoDB
        db.images.update_one({'img_id': 1}, {"$set": {"url": url}}, upsert=True)

        # Wait for 5 seconds
        time.sleep(5)

        # Optional: break loop after pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()