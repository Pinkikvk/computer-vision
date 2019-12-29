import cv2

class SmileDetector:
    def __init__(self, show_preview = True):
        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.smile_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')
        self.show_preview = show_preview
        self.previous_detections = []
    
    def detect(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_img, 1.3, 5)
        smile_detected = False
        for (x, y, w, h) in faces:
            face_gray_img = gray_img[y:y+h, w:w+h]
            smiles = self.smile_classifier.detectMultiScale(face_gray_img, 1.9, 20)
            cv2.rectangle(gray_img, (x, y), (x+w, y+h), 255, 1)
            for (sx, sy, sw, sh) in smiles:
                smile_detected = True
                cv2.rectangle(gray_img, (x + sx, y + sy), (x + sx + sw, y + sy + sh), 255, 1)

        if self.show_preview:
            cv2.imshow("Detector preview", gray_img)

        return self.isSmileDetected(smile_detected)

    def isSmileDetected(self, current_detection):
        self.previous_detections.append(int(current_detection))
        if len(self.previous_detections) > 10 :
            self.previous_detections.pop(0)

        sum = 0
        for previous_detection in self.previous_detections :
            sum += previous_detection

        return sum / len(self.previous_detections) > 0.5
