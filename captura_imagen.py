import cv2
import os
import tkinter as tk
from tkinter import Label, Button, filedialog
from PIL import Image, ImageTk
from deepface import DeepFace
import numpy as np

# Ruta donde se guardan las imágenes
folder_name = r"C:\\Users\\spard\\OneDrive\\Desktop\\AgeSense\\capturas"

# Crear carpeta si no existe
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Inicializar la cámara y el detector de rostros
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Crear ventana de la interfaz gráfica
root = tk.Tk()
root.title("Reconocimiento Facial con Estimación de Edad Mejorada")
root.geometry("800x600")
root.configure(bg="black")

# Etiquetas de la interfaz
label_video = Label(root, bg="black")
label_video.pack()

label_image = Label(root, bg="gray")
label_image.pack()

label_age = Label(root, text="Edad Estimada: -", font=("Arial", 16), fg="white", bg="black")
label_age.pack()

# Función para actualizar la vista de la cámara
def update_frame():
    global frame, img_tk
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        label_video.config(image=img_tk)
        label_video.image = img_tk

    label_video.after(10, update_frame)

# Función para capturar imagen y estimar edad
def capture_image():
    global frame
    if 'frame' in globals():
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w] 
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)  

            # Preprocesamiento de imagen
            roi = cv2.resize(roi, (224, 224))  # Redimensionar
            roi = cv2.equalizeHist(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))  # Mejorar contraste

            # Guardar imagen
            img_path = os.path.join(folder_name, "temp_face.jpg")
            cv2.imwrite(img_path, roi)

            # Analizar imagen con DeepFace con múltiples pruebas
            try:
                ages = []
                for _ in range(3):  # Realizar 3 análisis para mejorar precisión
                    analysis = DeepFace.analyze(img_path, actions=['age'], enforce_detection=False, model_name="VGG-Face")
                    ages.append(analysis[0]['age'])

                estimated_age = int(np.mean(ages))  # Promediar valores

                # Filtrar valores poco confiables
                if estimated_age < 5 or estimated_age > 90:
                    label_age.config(text="Detección poco confiable")
                else:
                    label_age.config(text=f"Edad Estimada: {estimated_age} años")

            except Exception as e:
                label_age.config(text="No se pudo estimar la edad")

# Función para seleccionar y analizar una imagen
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((500, 500))
        img_tk = ImageTk.PhotoImage(img)
        label_image.config(image=img_tk)
        label_image.image = img_tk

        # Analizar imagen con DeepFace
        try:
            ages = []
            for _ in range(3):  # Realizar 3 análisis para mejorar precisión
                analysis = DeepFace.analyze(file_path, actions=['age'], enforce_detection=False, model_name="VGG-Face")
                ages.append(analysis[0]['age'])

            estimated_age = int(np.mean(ages))

            # Filtrar valores poco confiables
            if estimated_age < 5 or estimated_age > 90:
                label_age.config(text="Detección poco confiable")
            else:
                label_age.config(text=f"Edad Estimada: {estimated_age} años")

        except Exception as e:
            label_age.config(text="No se pudo estimar la edad")

# Botones
btn_capture = Button(root, text="Capturar y Estimar Edad", command=capture_image, font=("Arial", 14), bg="green", fg="white")
btn_capture.pack(pady=5)

btn_select = Button(root, text="Seleccionar Imagen", command=select_image, font=("Arial", 14), bg="blue", fg="white")
btn_select.pack(pady=5)

# Iniciar la actualización del video
update_frame()

root.mainloop()
cap.release()
cv2.destroyAllWindows()
