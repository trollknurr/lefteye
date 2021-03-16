## Интро

Обертка для распознования объектов на основе darknet-сетей.
В пакете идет конфиг и веса для yolov4, но можно инициализировать класс любыми конфигами и весами для сетей этого класса.

## Зависимости

Пакет зависит только от OpenCV, причем нужен opencv-contrib. 
У колеса прописана зависимость от пакета `opencv-contrib-python-headless` для работы в серверных окружениях.

## Использование

На всякий случай: OpenCV не является thread-safe.

1) Инициализация:

```python
from lefteye import Detector
detector = Detector()
```

2) Детекция по матрице кадра в формате OpenCV:
```python
img = cv2.imread("%path-to-image%")  # или ret, image = cap.read() 
result = detector.detect_by_matrix(img)
```

3) Детекция по картинке используя Pillow:
```python
img = PIL.Image.open("%path-to-image%")
result = detector.detect_by_image(img)
```

