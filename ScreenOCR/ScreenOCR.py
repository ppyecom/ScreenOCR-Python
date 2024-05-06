
import cv2
import numpy as np
import pyautogui
import pytesseract

# Capturar toda la pantalla
captura = pyautogui.screenshot()

# Convertir la captura a una matriz numpy para mostrarla en OpenCV
captura_np = np.array(captura)
captura_cv2 = cv2.cvtColor(captura_np, cv2.COLOR_RGB2BGR)

# Crear una ventana de OpenCV y mostrar la captura de pantalla
cv2.imshow("Captura de pantalla", captura_cv2)

# Esperar a que el usuario seleccione una regi�n de la pantalla haciendo clic y arrastrando el mouse
rectangulo = cv2.selectROI("Captura de pantalla", captura_cv2, fromCenter=False, showCrosshair=True)

# Cerrar la ventana de OpenCV
cv2.destroyAllWindows()

# Obtener las coordenadas del rect�ngulo seleccionado
x, y, w, h = rectangulo

# Recortar la captura de pantalla original utilizando las coordenadas del rect�ngulo seleccionado
captura_recortada = captura.crop((x, y, x+w, y+h))

# Guardar la captura recortada como un archivo de imagen
captura_recortada.save("captura_recortada.png")

# Cargar la imagen de la captura recortada
captura_recortada = cv2.imread("captura_recortada.png")

# Convertir la imagen a escala de grises
captura_recortada_gris = cv2.cvtColor(captura_recortada, cv2.COLOR_BGR2GRAY)

# Realizar un preprocesamiento en la imagen para mejorar la calidad del reconocimiento
captura_recortada_preprocesada = cv2.threshold(captura_recortada_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Aplicar OCR a la imagen preprocesada y obtener el texto reconocido
texto_reconocido = pytesseract.image_to_string(captura_recortada_preprocesada, lang='spa', config='--psm 6')

# Imprimir el texto reconocido en la consola
print(texto_reconocido)