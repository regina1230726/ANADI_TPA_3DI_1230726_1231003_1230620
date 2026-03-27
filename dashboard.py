import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. Configuração da Página
st.set_page_config(page_title="Dashboard - Energia e VE", page_icon="⚡", layout="wide")

st.title("Dashboard de Energia e Mobilidade Elétrica")
st.markdown("Análise da libertação de potência na rede de Baixa Tensão através da modernização para tecnologia LED.")

# 2. Carregar os dados com Cache
@st.cache_data
def load_data():
    df = pd.read_csv("df_final.csv")
    ptd = pd.read_excel("PTD_data.xlsx")
    ip = pd.read_excel("IP_data.xlsx") # Lemos o ficheiro IP para ir buscar os Distritos
    
    # Preparar coordenadas para o mapa
    ptd[["lat", "lon"]] = ptd["Coordenadas Geográficas"].str.split(",", expand=True)
    ptd["lat"] = ptd["lat"].astype(float)
    ptd["lon"] = ptd["lon"].astype(float)
    
    # Criar mapas de nomes automáticos (Distrito e Concelho)
    mapa_concelhos = dict(zip(ip['CodDistritoConcelho'], ip['Concelho']))
    mapa_distritos = dict(zip(ip['CodDistritoConcelho'], ip['Distrito']))
    
    # Adicionar os nomes ao df_final
    df['NomeConcelho'] = df['CodDistritoConcelho'].map(mapa_concelhos).fillna(df['CodDistritoConcelho'].astype(str))
    df['NomeDistrito'] = df['CodDistritoConcelho'].map(mapa_distritos).fillna("Desconhecido")
    
    return df, ptd

df, ptd = load_data()

# -------------------------
# BARRA LATERAL (SIDEBAR) COM FILTROS EM CASCATA
# -------------------------
st.sidebar.header("Filtros de Análise")

# 1º Filtro: DISTRITO
distritos_disponiveis = sorted(df["NomeDistrito"].unique())
distritos_selecionados = st.sidebar.multiselect(
    "1. Selecionar Distrito(s)",
    options=distritos_disponiveis,
    help="Deixa vazio para ver todos os concelhos do país."
)

# LÓGICA DE CASCATA: Filtrar a lista de concelhos consoante o distrito escolhido
if distritos_selecionados:
    # Se escolheu distrito, mostra só os concelhos desse(s) distrito(s)
    concelhos_disponiveis = sorted(df[df["NomeDistrito"].isin(distritos_selecionados)]["NomeConcelho"].unique())
    concelhos_default = concelhos_disponiveis # Auto-seleciona todos os concelhos do distrito escolhido!
else:
    # Se não escolheu distrito, mostra tudo
    concelhos_disponiveis = sorted(df["NomeConcelho"].unique())
    concelhos_default = concelhos_disponiveis[:5] # Seleciona apenas 5 por defeito para não sobrecarregar

# 2º Filtro: CONCELHO
concelhos = st.sidebar.multiselect(
    "2. Selecionar Concelhos",
    options=concelhos_disponiveis,
    default=concelhos_default
)

st.sidebar.markdown("---")
st.sidebar.header("Simulação VE")
n_carregadores = st.sidebar.slider("Nº médio de carregadores (22kW) por PTD", 0, 50, 10)

# Filtrar o dataframe principal usando apenas os concelhos finais selecionados
df_filtrado = df[df["NomeConcelho"].isin(concelhos)].copy()

# Recálculos dinâmicos
df_filtrado["Depois_LED"] = df_filtrado["P_IP_Total"] - df_filtrado["Delta_PLED"]
df_filtrado["PVE_novo"] = df_filtrado["N_PTDs"] * 22 * (n_carregadores / 10)

# -------------------------
# MÉTRICAS DE TOPO (KPIs)
# -------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Potência IP Atual (kW)", f"{df_filtrado['P_IP_Total'].sum():,.0f}")
with col2:
    st.metric("Potência Libertada - LED (kW)", f"{df_filtrado['Delta_PLED'].sum():,.0f}", delta="Poupança")
with col3:
    st.metric("Capacidade PTD Instalada (kVA)", f"{df_filtrado['Cap_PTD'].sum():,.0f}")
with col4:
    st.metric("Folga Disponível na Rede (kW)", f"{df_filtrado['PFolga'].sum():,.0f}")

st.markdown("---")

# -------------------------
# SEPARADORES (TABS) PARA ORGANIZAÇÃO VISUAL
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Iluminação e Perfis", "⚡ Capacidade da Rede", "🚗 Impacto Veículos Elétricos", "🗺️ Mapa Georreferenciado"])

with tab1:
    st.subheader("Consumo de Iluminação: Antes vs Depois do LED")
    fig1 = px.bar(
        df_filtrado.head(15), 
        x="NomeConcelho", 
        y=["P_IP_Total", "Depois_LED"],
        barmode="group",
        labels={"value": "Potência (kW)", "variable": "Cenário", "NomeConcelho": "Concelho"},
        color_discrete_sequence=["#ef553b", "#00cc96"],
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Perfil Horário de Consumo (Simulação Diária)")
    horas = np.arange(24)
    perfil_base = np.where((horas >= 19) | (horas <= 7), 1, 0)
    
    # Verificação de segurança caso o dataframe fique vazio
    if len(df_filtrado) > 0:
        antes = perfil_base * df_filtrado["P_IP_Total"].mean()
        depois = perfil_base * df_filtrado["Depois_LED"].mean()
    else:
        antes = np.zeros(24)
        depois = np.zeros(24)
        
    df_perfil = pd.DataFrame({"Hora": horas, "Antes LED": antes, "Depois LED": depois})
    fig_perfil = px.area(
        df_perfil, x="Hora", y=["Antes LED", "Depois LED"],
        labels={"value": "Potência (kW)", "Hora": "Hora do Dia"},
        color_discrete_sequence=["#ef553b", "#00cc96"],
        template="plotly_white"
    )
    st.plotly_chart(fig_perfil, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Capacidade Instalada vs Folga")
        fig2 = px.bar(
            df_filtrado.head(15), x="NomeConcelho", y=["Cap_PTD", "PFolga"],
            barmode="group", template="plotly_white",
            labels={"value": "Potência", "variable": "Métrica", "NomeConcelho": "Concelho"},
            color_discrete_sequence=["#636efa", "#ab63fa"]
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_b:
        st.subheader("Top Potência Libertada (ΔPLED)")
        fig3 = px.bar(
            df_filtrado.sort_values("Delta_PLED", ascending=False).head(15),
            x="NomeConcelho", y="Delta_PLED",
            template="plotly_white", color="Delta_PLED",
            color_continuous_scale="Viridis",
            labels={"Delta_PLED": "Potência (kW)", "NomeConcelho": "Concelho"}
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Risco de Sobrecarga (Folga vs Consumo VE)")
        fig_ve = px.scatter(
            df_filtrado, x="PFolga", y="PVE_novo", size="Cap_PTD",
            color="D", color_continuous_scale="RdYlGn",
            hover_name="NomeConcelho", 
            labels={"PFolga": "Folga Existente (kW)", "PVE_novo": "Carga Projetada VE (kW)", "D": "Saldo Viabilidade"},
            template="plotly_white"
        )
        
        # Só adiciona a linha vermelha se houverem dados
        if len(df_filtrado) > 0:
            fig_ve.add_shape(type="line", x0=0, y0=0, x1=df_filtrado["PFolga"].max(), y1=df_filtrado["PFolga"].max(), line=dict(dash="dash", color="red"))
        
        st.plotly_chart(fig_ve, use_container_width=True)
    
    with col_d:
        st.subheader("Viabilidade Final (D) por Concelho")
        fig5 = px.bar(
            df_filtrado.sort_values("D", ascending=False).head(15),
            x="NomeConcelho", y="D",
            template="plotly_white", color="D",
            color_continuous_scale="RdYlGn",
            labels={"D": "Saldo Final (kW)", "NomeConcelho": "Concelho"}
        )
        st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.subheader("Localização da Infraestrutura (PTDs)")
    st.markdown("Postos de Transformação analisados no dataset. Aproxime o mapa para explorar as zonas.")
    
    # Filtra o mapa APENAS para os concelhos selecionados na barra lateral!
    ptd_filtrado = ptd[ptd["Concelho"].isin(concelhos)].dropna(subset=['lat', 'lon']).head(5000) 
    
    fig_map = px.scatter_map(
        ptd_filtrado, lat="lat", lon="lon",
        hover_name="Código de Instalação",
        hover_data=["Potência instalada [kVA]", "Concelho"],
        color="Nível de Utilização [%]", color_discrete_sequence=px.colors.qualitative.Pastel,
        zoom=5.5, height=600
    )
    fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)