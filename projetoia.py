import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image
from fpdf import FPDF

# Função para aplicar o filtro gaussiano com a constante sigmaX e combinar com a imagem original
def apply_gaussian_filter(image, sigmaX):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    result = cv2.addWeighted(image, 4, blurred, -4, 128)
    return result

# Função para prever a classe da imagem
def predict_class(image):
    temp_dir = './temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Limpar o diretório temporário
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Converter a imagem para o formato esperado
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Aplicar o filtro gaussiano
    sigmaX = 3
    filtered_img = apply_gaussian_filter(img_bgr, sigmaX)

    # Redimensionar a imagem
    resized_img = cv2.resize(filtered_img, (224, 224))

    # Normalizar a imagem para o modelo
    image = np.array(resized_img) / 255.0

    # Carregar o modelo e fazer a predição
    new_model = tf.keras.models.load_model("64x3-CNN.h5")
    predict = new_model.predict(np.array([image]))
    per = np.argmax(predict, axis=1)

    # Retornar os resultados
    diagnosis = (
        "Pré Diagnóstico: Não há Retinopatia Diabética"
        if per == 1
        else "Pré Diagnóstico: Possível Retinopatia Diabética"
    )
    return diagnosis, filtered_img

# Função para criar o PDF
def generate_pdf(original_image, filtered_image, diagnosis):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Relatório de Análise de Retinopatia Diabética", ln=True, align='C')
    pdf.ln(10)

    # Diagnóstico
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"Diagnóstico: {diagnosis}")
    pdf.ln(10)

    # Imagem original
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Imagem Original:", ln=True)
    original_path = "./temp/original_image.jpg"
    filtered_path = "./temp/filtered_image.jpg"

    original_image.save(original_path)
    pdf.image(original_path, x=10, y=50, w=90)

    # Imagem filtrada
    pdf.set_xy(110, 50)
    pdf.cell(0, 10, "Imagem Filtrada:", ln=True)
    cv2.imwrite(filtered_path, filtered_image)
    pdf.image(filtered_path, x=110, y=50, w=90)

    # Salvar PDF
    pdf_path = "./temp/Relatorio_Retinopatia.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Configuração do Streamlit
st.title("Detecção de Retinopatia Diabética")
st.write("Carregue uma imagem para realizar a análise.")

# Upload da imagem
uploaded_file = st.file_uploader("Selecione uma imagem", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Exibir a imagem original
    st.image(uploaded_file, caption="Imagem Original", use_column_width=True)

    # Abrir a imagem com PIL
    image = Image.open(uploaded_file)

    # Realizar a predição
    with st.spinner("Processando..."):
        diagnosis, filtered_image = predict_class(image)

    # Exibir os resultados
    st.image(filtered_image, caption="Imagem Filtrada e Redimensionada", use_column_width=True)
    st.success(diagnosis)

    # Gerar o PDF
    with st.spinner("Gerando PDF..."):
        pdf_path = generate_pdf(image, filtered_image, diagnosis)

    # Disponibilizar o download do PDF
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="Baixar Relatório em PDF",
            data=pdf_file,
            file_name="Relatorio_Retinopatia.pdf",
            mime="application/pdf",
        )
