import cv2
import os

harcascade = r"D:\virtual studio code\project using python\face emotion detection\Vehical Number Plate\haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)


cap.set(3, 640)  
cap.set(4, 480) 

min_area = 500


count = 0


output_dir = "plates"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


while True:
    success, img = cap.read()

    
    if not success:
        print("Failed to capture video. Exiting...")
        break

    
    plate_cascade = cv2.CascadeClassifier(harcascade)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x: x + w]
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        if len(plates) > 0:
            plate_img_path = os.path.join(output_dir, f"scanned_img_{count}.jpg")
            cv2.imwrite(plate_img_path, img_roi)
            
            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(500)

            count += 1
        else:
            print("No plate detected, unable to save.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
