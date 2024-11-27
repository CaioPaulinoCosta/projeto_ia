import streamlit as st
import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from fpdf import FPDF
import tensorflow as tf

# Configuração do Streamlit
st.title("Cadastro de Paciente e Detecção de Retinopatia Diabética")

# Inicializar o estado do Streamlit
if "paciente_cadastrado" not in st.session_state:
    st.session_state.paciente_cadastrado = False
    st.session_state.uploaded_image_path = None

# Cadastro do paciente
st.subheader("Cadastro do Paciente")

# Coletar informações do paciente
nome = st.text_input("Nome Completo")
idade = st.number_input("Idade", min_value=0, max_value=120)
sexo = st.selectbox("Sexo", ["Masculino", "Feminino", "Outro"])
telefone = st.text_input("Telefone")
email = st.text_input("E-mail")
upload_file = st.file_uploader("Selecione uma imagem", type=["png", "jpg", "jpeg"])

# Pasta de upload (criar se não existir)
upload_dir = './uploads'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

if st.button("Cadastrar Paciente"):
    if nome and idade and sexo and telefone and email and upload_file:
        # Exibir as informações do paciente cadastradas em uma tabela
        patient_data = {
            "Nome": [nome],
            "Idade": [idade],
            "Sexo": [sexo],
            "Telefone": [telefone],
            "E-mail": [email]
        }
        patient_df = pd.DataFrame(patient_data)
        st.table(patient_df)

        # Salvar a imagem enviada na pasta 'uploads'
        image_path = os.path.join(upload_dir, f"{nome}_imagem.jpg")
        try:
            image = Image.open(upload_file)
            image.save(image_path)
            st.session_state.uploaded_image_path = image_path
            st.session_state.paciente_cadastrado = True

            # Exibir a imagem carregada
            st.image(image, caption="Imagem do Paciente", use_container_width=True)
            st.success(f"Paciente {nome} cadastrado com sucesso!")
        except Exception as e:
            st.error("Erro ao salvar ou processar a imagem.")
            st.error(f"Detalhes do erro: {e}")
    else:
        st.error("Por favor, preencha todos os campos e envie uma imagem.")

# Exibir opções de análise apenas após o cadastro do paciente
if st.session_state.paciente_cadastrado:
    st.subheader("Análise da Retinopatia Diabética")
    if st.button("Fazer Análise"):
        image_path = st.session_state.uploaded_image_path
        if image_path:
            uploaded_file = Image.open(image_path)

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
                new_model = tf.keras.models.load_model("64x3-CNN.keras")
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

            # Realizar a predição
            with st.spinner("Processando..."):
                diagnosis, filtered_image = predict_class(uploaded_file)

            # Exibir os resultados
            st.image(filtered_image, caption="Imagem Filtrada e Redimensionada", use_container_width=True)
            st.success(diagnosis)

            # Gerar o PDF
            with st.spinner("Gerando PDF..."):
                pdf_path = generate_pdf(uploaded_file, filtered_image, diagnosis)

            # Disponibilizar o download do PDF
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Baixar Relatório em PDF",
                    data=pdf_file,
                    file_name="Relatorio_Retinopatia.pdf",
                    mime="application/pdf",
                )
