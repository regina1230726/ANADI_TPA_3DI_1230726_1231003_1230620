import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Dashboard - Energia e Mobilidade Elétrica")

df = pd.read_csv("df_final.csv")

# -------------------------
# FILTRO
# -------------------------
concelhos = st.multiselect(
    "Selecionar Concelhos",
    df["CodDistritoConcelho"].astype(str),
    default=df["CodDistritoConcelho"].astype(str)
)

df_filtrado = df[df["CodDistritoConcelho"].astype(str).isin(concelhos)]

# -------------------------
# 1. PERFIS (ANTES vs DEPOIS)
# -------------------------
st.subheader("Perfis de Consumo (Antes vs Depois LED)")

df_filtrado = df_filtrado.copy()
df_filtrado["Depois_LED"] = df_filtrado["P_IP_Total"] - df_filtrado["Delta_PLED"]

fig1 = px.bar(
    df_filtrado.head(10),
    x="CodDistritoConcelho",
    y=["P_IP_Total", "Depois_LED"],
    barmode="group",
    title="Consumo Antes vs Após Implementação LED"
)

st.plotly_chart(fig1)

# -------------------------
# 2. CAPACIDADE INSTALADA vs DISPONÍVEL
# -------------------------
st.subheader("Capacidade da Rede")

fig2 = px.bar(
    df_filtrado.head(10),
    x="CodDistritoConcelho",
    y=["Cap_PTD", "PFolga"],
    barmode="group",
    title="Capacidade Instalada vs Disponível"
)

st.plotly_chart(fig2)

# -------------------------
# 3. POTÊNCIA LIBERTADA
# -------------------------
st.subheader("Potência Libertada (ΔPLED)")

fig3 = px.bar(
    df_filtrado.sort_values("Delta_PLED", ascending=False).head(10),
    x="CodDistritoConcelho",
    y="Delta_PLED",
    title="Top Concelhos com Maior Potência Libertada"
)

st.plotly_chart(fig3)

# -------------------------
# 4. CENÁRIO VE (IMPACTO)
# -------------------------

st.subheader("Cenário de Carregadores VE")

n_carregadores = st.slider("Número médio de carregadores por PTD", 0, 50, 10)

df_filtrado["PVE_novo"] = df_filtrado["N_PTDs"] * 22 * (n_carregadores / 10)

fig_ve = px.scatter(
    df_filtrado,
    x="PFolga",
    y="PVE_novo",
    size="D",
    title="Impacto de VE na Rede"
)

fig_ve.add_shape(
    type="line",
    x0=0, y0=0,
    x1=df_filtrado["PFolga"].max(),
    y1=df_filtrado["PFolga"].max(),
    line=dict(dash="dash")
)

st.plotly_chart(fig_ve)

# -------------------------
# 5. VIABILIDADE FINAL
# -------------------------
st.subheader("Viabilidade Final (D)")

fig5 = px.bar(
    df_filtrado.sort_values("D", ascending=False).head(10),
    x="CodDistritoConcelho",
    y="D",
    title="Top Concelhos Mais Viáveis"
)

st.plotly_chart(fig5)

# -------------------------
# 6. MAPA DOS PTDs
# -------------------------
st.subheader("Mapa dos PTDs")

# carregar ficheiro PTD com coordenadas
ptd = pd.read_excel("PTD_data.xlsx")

# separar latitude e longitude
ptd[["lat", "lon"]] = ptd["Coordenadas Geográficas"].str.split(",", expand=True)
ptd["lat"] = ptd["lat"].astype(float)
ptd["lon"] = ptd["lon"].astype(float)

# mapa
fig_map = px.scatter_map(
    ptd,
    lat="lat",
    lon="lon",
    hover_name="Código de Instalação",
    hover_data=["Potência instalada [kVA]", "Concelho"],
    zoom=6,
    height=500
)

fig_map.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig_map)

# -------------------------
# PERFIL HORÁRIO SIMULADO
# -------------------------
st.subheader("Perfil Horário (Simulado)")

import numpy as np

horas = np.arange(24)

# perfil típico iluminação pública (liga à noite)
perfil_base = np.where((horas >= 19) | (horas <= 7), 1, 0)

antes = perfil_base * df_filtrado["P_IP_Total"].mean()
depois = perfil_base * df_filtrado["Depois_LED"].mean()

df_perfil = pd.DataFrame({
    "Hora": horas,
    "Antes": antes,
    "Depois": depois
})

fig_perfil = px.line(
    df_perfil,
    x="Hora",
    y=["Antes", "Depois"],
    title="Perfil Horário (Simulado)"
)

st.plotly_chart(fig_perfil)