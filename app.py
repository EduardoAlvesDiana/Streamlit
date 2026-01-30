import streamlit as st
import pandas as pd
import joblib

# Carregar modelo e transformadores
model = joblib.load('modelo_defasagem_lr.pkl')
scaler = joblib.load('scaler_defasagem.pkl')
imputer = joblib.load('imputer_defasagem.pkl')

st.set_page_config(page_title="Previsão de Defasagem Escolar", layout="wide")
st.title("📚 Previsão de Risco de Defasagem Escolar")
st.write("Responda às perguntas abaixo para prever se o aluno está em risco de defasagem escolar.")

# Formulário de entrada
Genero = st.selectbox("Qual é o gênero do aluno?", ["Feminino", "Masculino"])
Idade = st.slider("Qual é a idade do aluno?", 10, 30, 16)
Data_de_Nasc = st.number_input("Qual é o ano de nascimento do aluno?", min_value=1990, max_value=2020, value=2006)
Ano_ingresso = st.number_input("Em que ano o aluno ingressou?", min_value=2000, max_value=2026, value=2018)
Inst_de_ensino = st.selectbox("Qual é a instituição de ensino?", ["Privada", "Publica"])
Ano_Pesquisa = st.number_input("Qual é o ano da pesquisa?", min_value=2022, max_value=2026, value=2024)

INDE = st.slider("Qual é o índice de desenvolvimento educacional (INDE)?", 0.0, 10.0, 6.5)
IAA = st.slider("Qual é o indicador de adequação acadêmica (IAA)?", 0.0, 10.0, 7.5)
IEG = st.slider("Qual é o indicador de engajamento (IEG)?", 0.0, 10.0, 6.0)
IPS = st.slider("Qual é o indicador psicossocial (IPS)?", 0.0, 10.0, 5.8)
IDA = st.slider("Qual é o indicador de desempenho acadêmico (IDA)?", 0.0, 10.0, 6.2)
IPV = st.slider("Qual é o indicador de participação e vivência (IPV)?", 0.0, 10.0, 7.0)
IAN = st.slider("Qual é o indicador de adequação ao nível (IAN)?", 0.0, 10.0, 8.0)

# Botão de previsão
if st.button("Prever risco de defasagem"):
    # Montar DataFrame
    input_dict = {
        "INDE": INDE,
        "Data_de_Nasc": Data_de_Nasc,
        "Idade": Idade,
        "Genero": 0 if Genero == "Feminino" else 1,
        "Ano_ingresso": Ano_ingresso,
        "Inst_de_ensino": 1 if Inst_de_ensino == "Privada" else 0,
        "IAA": round(IAA, 1),
        "IEG": round(IEG, 1),
        "IPS": round(IPS, 1),
        "IDA": round(IDA, 1),
        "IPV": round(IPV, 1),
        "IAN": round(IAN, 1),
        "Ano_Pesquisa": Ano_Pesquisa
    }

    aluno_df = pd.DataFrame([input_dict])

    # 🔑 Ajuste da opção 1: garantir que todas as colunas sejam numéricas
    aluno_df = aluno_df.astype(float)

    # Transformações
    aluno_imputed = imputer.transform(aluno_df)
    aluno_scaled = scaler.transform(aluno_imputed)

    # Previsão
    prob = model.predict_proba(aluno_scaled)[0][1]
    pred = model.predict(aluno_scaled)[0]

    st.subheader("🔍 Resultado da Previsão")
    st.metric("Probabilidade de risco de defasagem", f"{prob:.2%}")
    if pred == 0:
        st.success("🟩 O aluno está **sem risco de defasagem**.")
    else:
        st.error("🟥 O aluno está **em risco de defasagem**.")
