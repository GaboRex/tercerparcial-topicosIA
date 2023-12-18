from mediapipe_model_maker import gesture_recognizer
import numpy as np
import cv2

# Carga el modelo desde el archivo .task
model = gesture_recognizer.GestureRecognitionModel.load("ruta_del_modelo.task")

# Inicializa la cámara
cap = cv2.VideoCapture(0)

while True:
    # Captura el video desde la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Convierte el marco a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Realiza la inferencia en el marco
    input_data = gesture_recognizer.HandDataPreprocessingParams().mediapipe_model_input_transform(gray)
    predictions = model.predict(np.expand_dims(input_data, axis=0))

    # Obtiene la etiqueta de la clase predicha
    predicted_class = np.argmax(predictions)

    # Dibuja el resultado en el marco
    cv2.putText(frame, f"Predicted Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Muestra el marco resultante
    cv2.imshow('Hand Gesture Recognition', frame)

    # Sale del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos
cap.release()
cv2.destroyAllWindows()
