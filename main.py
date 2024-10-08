from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import easyocr
import csv
import requests
import time

reader = easyocr.Reader(['en'])
license_plate_detector = YOLO('license_plate_detector.pt')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
api_url = "https://fcm-signalr-min.azurewebsites.net/api/handleRegNumber"

csv_filename = 'detected_plates_log.csv'

with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame Number', 'Plate Text', 'Timestamp'])

    frame_nmr = -1
    ret = True
    last_request_time = time.time()
    request_interval = 2  # Send a request at most once every 1 second

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()

        if ret:
            
            license_plates = license_plate_detector(frame)[0]

            if not license_plates.boxes:
                print("No Plate Detected")
            else:
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    license_plate_crop_contrast = cv2.convertScaleAbs(license_plate_crop_gray, alpha=1.5, beta=20)
                    license_plate_crop_blur = cv2.GaussianBlur(license_plate_crop_contrast, (5, 5), 0)
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    license_plate_crop_sharp = cv2.filter2D(license_plate_crop_blur, -1, kernel)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    ocr_results = reader.readtext(license_plate_crop_thresh)

                    if ocr_results and (time.time() - last_request_time) >= request_interval:
                        for detection in ocr_results:
                            plate_text = detection[1]
                            print(f"Detected Plate: {plate_text}")

                            try:
                                response = requests.post(api_url, json={"regnumber": plate_text})
                                if response.status_code == 200:
                                    print(f"POST successful: {plate_text}")
                                else:
                                    print(f"Failed to POST: {response.status_code}, {response.text}")
                                last_request_time = time.time()
                            except Exception as e:
                                print(f"Error during POST: {e}")

                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame, plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            csv_writer.writerow([frame_nmr, plate_text, timestamp])
                    else:
                        print("OCR failed to detect plate text.")

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()