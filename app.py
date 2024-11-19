# Sistema de Reconocimiento Visual de Señales de tráfico mediante Redes Neuronales Convolucionales

import io
import time

import pathlib
from PIL import Image
import tensorflow as tf

from tensorflow import keras

from keras import layers
from keras import Sequential
from keras import Model
import numpy as np

import cv2
import torch

from ultralytics.utils.checks import check_requirements


chanDim = -1
altura_img = 32
anchura_img = 32

modelo = Sequential([
    layers.Rescaling(1./255, input_shape=(altura_img, anchura_img, 3)),

    # CONV => RELU => BN => POOL
    layers.Conv2D(8, (5,5), activation='relu',padding='same'),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(2,2),

     # first set of (CONV => RELU => CONV => RELU) * 2 => POOL
    layers.Conv2D(16, (3,3), activation='relu',padding='same'),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(16, (3,3), activation='relu',padding='same'),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(2,2),

    # second set of (CONV => RELU => CONV => RELU) * 2 => POOL
     layers.Conv2D(32, (3,3), activation='relu',padding='same'),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu',padding='same'),
    layers.BatchNormalization(axis=chanDim),
    layers.MaxPooling2D(2,2),

  # first set of FC => RELU layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

 		# second set of FC => RELU layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

	# Classifier
    layers.Dense(43),
])


mystyle = '''
    <style>
        p {
            text-align: justify;
        }
    </style>
    '''

data_dir ="Signals"
data_dir = pathlib.Path(data_dir)

nombre_clases = ['0_Limite_velocidad_(20kmh)',
 '10_Prohibido_adelantar_para_camiones',
 '11_Interseccion_con _prioridad',
 '12_Calzada_prioridad',
 '13_Ceda_el_paso',
 '14_Stop',
 '15_Circulacion_prohibida',
 '16_Entrada_prohibida_a_vehiculos_de_mercancias',
 '17_Prohibido',
 '18_Otros_peligros',
 '19_Curva _peligrosa_izquierda',
 '1_Limite_velocidad_(30kmh)',
 '20_Curva _peligrosa_derecha',
 '21_Curvas_Peligrosas',
 '22_Suelo_irregular',
 '23_Pavimento_deslizante',
 '24_Estrechamiento_de_la_calzada',
 '25_Obras',
 '26_Tramo_semaforos',
 '27_Paso_de_peatones',
 '28_Cruce_Niños',
 '29_Cruce_ciclistas',
 '2_Limite_velocidad_(50kmh)',
 '30_Riesgo_de_Hielo',
 '31_Animales_Salvajes',
 '32_Fin_de_prohibiciones',
 '33_Sentido_obligatorio_derecha',
 '34_Sentido_obligatorio_izquierda',
 '35_Sentido_obligatorio_recto',
 '36_Sentido_obligatorio_derecha recto',
 '37_Sentido_obligatorio_izquierda recto',
 '38_Paso_obligatorio_derecha',
 '39_Paso_obligatorio_izquierda',
 '3_Limite_velocidad_(60kmh)',
 '40_Sentidp_giratorio_obligatorio',
 '41_Fin_de_no_adelantar',
 '42_Fin_de_no_adelantar_para_camiones',
 '4_Limite_velocidad_(70kmh)',
 '5_Limite_velocidad_(80kmh)',
 '6_Fin_Limite_velocidad_(80kmh)',
 '7_Limite_velocidad_(100kmh)',
 '8_Limite_velocidad_(120kmh)',
 '9_Prohibido_adelantar']



def inference(model=None):
    """Sistema de Reconocimiento Visual de Señales de tráfico mediante Redes Neuronales Convolucionales."""
    check_requirements("streamlit>=1.29.0")  # scope imports for faster  package load speeds
    import streamlit as st

    from ultralytics import YOLO

    # Hide main menu style
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

    # Main title of streamlit application
    main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; 
                             font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                    Trabajo de Fin de Grado Ingenieria Informatica Online UEM
                    </h1></div>"""

    # Subtitle of streamlit application
    sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                    Sistema de Reconocimiento Visual de Señales de tráfico mediante Redes Neuronales Convolucionales</h4>
                    </div>"""
    
    # Subtitle 2 of streamlit application
    sub_title2_cfg = """<div><h6 style="color:white; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                    El objetivo principal de este proyecto es desarrollar un sistema de ayuda a la conducción
                    que permita clasificar señales de tráfico a partir de imágenes capturadas por una cámara.
                    Este sistema se basará en técnicas de Deep Learning, específicamente en el uso de redes neuronales
                    convolucionales (CNNs), para lograr un alto nivel de precisión y robustez en la clasificación</h6>
                    </div>"""
    
    # Subtitle 3 of streamlit application
    sub_title3_cfg = """<div><h6 style="color:white; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;"> 
                                       
      Antes de usar la cámara:

      Asegúrate de que tu cámara frontal  esté limpia y libre de obstrucciones.

      Cómo usar la cámara:

      Activa la camara.

      Apunta con la camara a la señal de trafico.
      La cámara detectará las señales de tráfico y mostrará la información.

      Limitaciones de las cámaras que detectan señales de tráfico:

        1.El sistema puede no detectar todas las señales de tráfico.

        2.El sistema puede ser menos preciso en condiciones climáticas adversas.
        """

    # Set html page configuration
    st.set_page_config(page_title="TrafficSigns Detec App", layout="wide", initial_sidebar_state="auto")

    # Append the custom HTML
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title2_cfg, unsafe_allow_html=True)
    st.markdown(sub_title3_cfg, unsafe_allow_html=True)

    # Add  logo in sidebar
    with st.sidebar:
        logo = "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcSizJ2YekDM7Wm2yUCVn6xIpe-rRkgosJMaoRd2yR4q9WLajqZu"
        st.image(logo, width=250)

    # Add elements to vertical setting menu
    st.sidebar.title("User Configuration")

    # Add video source selection dropdown
    source = st.sidebar.selectbox(
        "Modo",
        ("webcam", "video", "foto"),
    )

    vid_file_name = ""
    if source == "video":
        vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        if vid_file is not None:
            g = io.BytesIO(vid_file.read())  # BytesIO Object
            vid_location = "video.mp4"
            with open(vid_location, "wb") as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file
            vid_file_name = "video.mp4"
    elif source == "webcam":
        vid_file_name = 2
    elif source == "foto":

       # Recargue el modelo de los archivos que guardamos       
        modelo.load_weights('path_to_my_weights.h5')
        
        img_file_buffer = st.camera_input("Foto")

        if img_file_buffer is not None:
            # To read image file buffer as a PIL Image:
            img = Image.open(img_file_buffer)
            new_image = img.resize((32,32))

            # To convert PIL Image to numpy array:
            img_array = np.array(img)
            new_img_array = np.array(new_image)



            new_img_array = tf.expand_dims(new_img_array, 0)
            prediccion = modelo.predict(new_img_array)

            puntuacion = tf.nn.softmax(prediccion[0])

            prueba = nombre_clases[np.argmax(puntuacion)]
            Traffic_signal = list(data_dir.glob(prueba+'/*.png'))
            Traffic_signal2 = list(data_dir.glob(prueba+'/*.txt'))
            
            imagen2 = Image.open(str(Traffic_signal[0]))
            archivo = open(str(Traffic_signal2[0]),'r',encoding="utf8")
            st.image(imagen2)

            st.write("Esta imagen es de una señal de  {} con una confianza del {:.2f} por ciento."

            .format(nombre_clases[np.argmax(puntuacion)], 100 * np.max(puntuacion)))

            st.write(archivo.read())
       

    # Add dropdown menu for model selection
    available_models = ["best.pt"]
    if model:
        available_models.insert(0, model.split("best.pt")[0])  # insert model 

    selected_model = (available_models)
    with st.spinner("Model is downloading..."):
        model = YOLO("best.pt")  # Load the YOLO model
        class_names = list(model.names.values())  # Convert dictionary to list of class names
  


    conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.01))
    iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.55, 0.01))
    
    enable_trk = st.sidebar.radio("Enable Tracking", ("Yes", "No"))

    col1, col2 = st.columns(2)
    org_frame = col1.empty()
    ann_frame = col2.empty()

    fps_display = st.sidebar.empty()  # Placeholder for FPS display

    
    # Multiselect box with class names and get indices of selected classes
    selected_classes = st.sidebar.multiselect("Classes", class_names, default=class_names[:43])
    selected_ind = [class_names.index(option) for option in selected_classes]

    if not isinstance(selected_ind, list):  # Ensure selected_options is a list
        selected_ind = list(selected_ind)

    if st.button("Start", type="primary"):
        videocapture = cv2.VideoCapture(vid_file_name)  # Capture the video

        if not videocapture.isOpened():
            st.error("Could not open webcam.")

        stop_button = st.button("Stop")  # Button to stop the inference

        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                st.warning("Failed to read frame from webcam. Please make sure the webcam is connected properly.")
                break

            prev_time = time.time()  # Store initial time for FPS calculation

            # Store model predictions
            if enable_trk == "Yes":
                results = model.track(frame, conf=conf, iou=iou, classes=selected_ind, persist=True)
            else:
                results = model(frame, conf=conf, iou=iou, classes=selected_ind)

            annotated_frame = results[0].plot()  # Add annotations on frame

            # Calculate model FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)

            # display frame
            org_frame.image(frame, channels="BGR")
            ann_frame.image(annotated_frame, channels="BGR")

            if stop_button:
                videocapture.release()  # Release the capture
                torch.cuda.empty_cache()  # Clear CUDA memory
                st.stop()  # Stop streamlit app

            # Display FPS in sidebar
            fps_display.metric("FPS", f"{fps:.2f}")

        # Release the capture
        videocapture.release()

    # Clear CUDA memory
    torch.cuda.empty_cache()

    # Destroy window
    cv2.destroyAllWindows()


# Main function call
if __name__ == "__main__":
    inference()
