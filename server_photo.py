from flask import Flask, request
import datetime
import os
import cv2
import pytesseract
from photo_ocr import load_image, ocr, detection, recognition, draw_ocr, draw_detections

app = Flask(__name__)

# Папка для сохранения фото
UPLOAD_FOLDER = 'uploads'
INPUT_FOLDER = r'C:\Users\AlexF\Desktop\SERVER_PHOTO\uploads'
OUTPUT_PATH = r'C:\Users\AlexF\Desktop\SERVER_PHOTO\detected'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_text():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Используем Tesseract для распознавания текста
        text = pytesseract.image_to_string(gray, lang='rus+eng')
        
        # Получаем информацию о текстовых блоках
        data = pytesseract.image_to_data(gray, lang='rus+eng', output_type=pytesseract.Output.DICT)
        
        # Рисуем прямоугольники вокруг текста
        for i in range(len(data['level'])):
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Показываем результат
        cv2.imshow('Detect text', image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


def extract_text_from_image(photo, lang='rus+eng'):
    # Загрузка изображения
    image = cv2.imread(INPUT_FOLDER+'\\'+photo)
    if image is None:
        print("Ошибка чтения изображения")
        return None
    # Конвертация в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применение threshold для улучшения контраста
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Распознавание текста
    try:
        text = pytesseract.image_to_string(thresh, lang=lang)
        if text:
            print(text.strip())
    except Exception as e:
        print(f"Ошибка распознавания: {e}")
    cv2.imshow('Text', image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

def detect_face(photo):
    face_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
    frame = cv2.imread(INPUT_FOLDER+'\\'+photo)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
  
    print(f"Обнаружено {len(faces)} лиц")
    if(len(faces) > 0):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        output_path = OUTPUT_PATH+'\\'+photo
        cv2.imwrite(output_path, frame)
        print(f"Фото сохранено {output_path}")
        cv2.imshow('Detected Faces', frame)
    else:
        cv2.imshow('Dont detected Faces', frame)
    
    # Показать результат
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image/jpeg' in request.content_type:
        # Генерируем имя файла с временной меткой
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Сохраняем фото
        with open(filepath, 'wb') as f:
            f.write(request.data)
        
        print(f"Photo saved: {filename}")

        # detect_face(filename)
        extract_text_from_image(filename, 'rus+eng')
        return "Photo received successfully", 200
    else:
        return "Invalid content type", 400

if __name__ == '__main__':
    print("Server started. Waiting for photos...")
    # app.run(host='0.0.0.0', port=8080, debug=True)
    extract_text()