import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import ttest_ind
from scipy.stats import kurtosis
import scipy.stats as stats


# 4.2 MANIPULAÇÃO DE DADOS

# 4.2.1. Ficheiro IP

print("\n--- 4.2.1: Processamento da Iluminação Pública ---")

# 0. Ler o ficheiro de dados
# (Ajusta o nome do ficheiro para .xlsx ou .csv conforme o que estiveres a usar no teu código)
ip = pd.read_excel("IP_data.xlsx") 

# 1. Criar variável binária Is_Ineficiente
ip["Is_Ineficiente"] = ip["Tipo de Lâmpada"].isin(["Sódio", "Mercúrio"]).astype(int)

# 2. Criar variável Potência kW
# Atenção ao nome real da coluna no ficheiro: Potência Instalada Total (W)
ip["Potência kW"] = ip["Potência Instalada Total (W)"] / 1000

# Coluna auxiliar para calcular a potência ineficiente mais facilmente no groupby
ip["Potencia_Inef_Temp"] = ip["Potência kW"] * ip["Is_Ineficiente"]

# 3. Agrupar por CodDistritoConcelho
ip_group = ip.groupby("CodDistritoConcelho").agg(
    P_IP_Total=("Potência kW", "sum"),
    P_IP_Inef=("Potencia_Inef_Temp", "sum")
).reset_index()

# Opcional: Apagar a coluna temporária do dataframe original (boas práticas)
ip = ip.drop(columns=["Potencia_Inef_Temp"])

print("Resultado do agrupamento IP (Primeiras linhas):")
print(ip_group.head())

# 4.2.2. Ficheiro PTD

def converter_utilizacao(valor):

    if pd.isna(valor):
        return np.nan

    valor = str(valor).strip()

    # casos tipo "60%-79%"
    if "-" in valor and "%" in valor:
        return float(valor.split("-")[1].replace("%", "")) / 100

    # caso "+100%"
    if "+" in valor:
        return 1.0

    # caso "<20"
    if "<" in valor:
        return 0.20

    # caso "N/D"
    if "N/D" in valor:
        return np.nan

    return np.nan

ptd = pd.read_excel("PTD_data.xlsx")

ptd["Utilizacao_decimal"] = ptd["Nível de Utilização [%]"].apply(converter_utilizacao)

print("4.2.2: Conversão de nível de utilização para decimal:")
print(ptd[["Nível de Utilização [%]", "Utilizacao_decimal"]].head())

ptd_group = ptd.groupby("CodDistritoConcelho").agg(

    Cap_PTD=("Potência instalada [kVA]", "sum"),

    Util_Media=("Utilizacao_decimal", "mean"),

    N_PTDs=("Código de Instalação", "count")

).reset_index()

print()
print("4.2.2: Agrupar por Código Distrito Concelho")
print(ptd_group.head())

 # 4.2.3

df_final = pd.merge(ip_group, ptd_group, on="CodDistritoConcelho", how="inner")

df_final["Delta_PLED"] = df_final["P_IP_Inef"] * 0.65
df_final["PFolga"] = (df_final["Cap_PTD"] * 0.92) * (1 - df_final["Util_Media"])
df_final["PVE"] = df_final["N_PTDs"] * 22 * 0.60
df_final["D"] = df_final["PFolga"] + df_final["Delta_PLED"] - df_final["PVE"]
df_final["Rate_Ineficiencia"] = df_final["P_IP_Inef"] / df_final["P_IP_Total"]


# criar tabela com nomes mais intuitivos
tabela_resumo = df_final[[
    "CodDistritoConcelho",
    "P_IP_Total",
    "P_IP_Inef",
    "Cap_PTD",
    "Util_Media",
    "N_PTDs",
    "Delta_PLED",
    "PFolga",
    "PVE",
    "D",
    "Rate_Ineficiencia"
]].copy()

# renomear colunas para o relatório
tabela_resumo.columns = [
    "CodDistritoConcelho",
    "P_IP_Total (kW)",
    "P_IP_Inef (kW)",
    "Capacidade PTD (kVA)",
    "Utilização Média",
    "Nº PTDs",
    "Ganho LED (kW)",
    "Folga Rede (kW)",
    "Carga VE (kW)",
    "Saldo Final D (kW)",
    "Taxa Ineficiência"
]

# arredondar valores para ficar mais apresentável
tabela_resumo = tabela_resumo.round(4)

# mostrar apenas 10 concelhos
pd.set_option('display.max_columns', None)
print("\n--- 4.2.3: Tabela resumo (10 concelhos) ---")
print(tabela_resumo.head(10))

# 4.3 ANÁLISE E EXPLORAÇÃO DE DADOS

# 4.3.1 - Mix tecnológico (LED vs Convencional)

print("\n--- 4.3.1: Mix tecnológico da iluminação pública ---")

# classificar tecnologia
ip["Tecnologia"] = ip["Is_Ineficiente"].map({
    1: "Convencional (Sódio/Mercúrio)",
    0: "LED / Outras eficientes"
})

# somar potência por tecnologia
mix_tecnologico = ip.groupby("Tecnologia")["Potência kW"].sum()

print("Potência total por tecnologia:")
print(mix_tecnologico)

# gráfico pie
plt.figure(figsize=(6,6))
mix_tecnologico.plot(kind="pie", autopct="%1.1f%%")

plt.title("Mix Tecnológico da Iluminação Pública")
plt.ylabel("")

plt.show()


# verificar concentração da potência ineficiente por município
top_ineficientes = ip_group.sort_values("P_IP_Inef", ascending=False).head(10)

print()
print("Concelhos com maior potência ineficiente:")
print(top_ineficientes[["CodDistritoConcelho", "P_IP_Inef"]])

plt.figure(figsize=(10,6))
plt.bar(top_ineficientes["CodDistritoConcelho"].astype(str), top_ineficientes["P_IP_Inef"])

plt.title("Concelhos com Maior Potência de Iluminação Ineficiente")
plt.xlabel("Concelho")
plt.ylabel("Potência Ineficiente (kW)")

plt.xticks(rotation=45)

plt.show()

# 4.3.2 - Boxplots por Distrito

print("\n--- 4.3.2: Boxplots de Utilização por Distrito ---")

# 1. Extrair o código do Distrito (os primeiros dígitos do CodDistritoConcelho)
# Como o CodDistritoConcelho tem 3 ou 4 dígitos, uma divisão inteira por 100 dá-nos o Distrito.
ptd["CodDistrito"] = ptd["CodDistritoConcelho"] // 100

# 2. Mapear o código para o nome do Distrito correspondente
mapa_distritos = {
    1: "Aveiro",
    11: "Lisboa",
    13: "Porto",
    15: "Setúbal"
}
ptd["Distrito"] = ptd["CodDistrito"].map(mapa_distritos)

# 3. Definir os distritos que queremos analisar e remover valores nulos
distritos_alvo = ["Lisboa", "Porto", "Aveiro", "Setúbal"]
ptd_filtrado = ptd[ptd["Distrito"].isin(distritos_alvo)].dropna(subset=["Utilizacao_decimal"])

# 4. Criar a caixa de bigodes (boxplot)
plt.figure(figsize=(10, 6))
ptd_filtrado.boxplot(column="Utilizacao_decimal", by="Distrito", grid=False)

# Formatar o gráfico
plt.title("Distribuição do Nível de Utilização dos PTDs por Distrito")
plt.suptitle("") # Remove o subtítulo automático
plt.xlabel("Distrito")
plt.ylabel("Nível de Utilização (Decimal)")

# Mostrar o gráfico (não te esqueças de guardar ou fazer print screen para o relatório!)
plt.show()

# 5. Calcular o desvio padrão para responder à pergunta "maior variabilidade"
variabilidade = ptd_filtrado.groupby("Distrito")["Utilizacao_decimal"].std().sort_values(ascending=False)
print("Variabilidade (Desvio Padrão) da utilização por distrito:")
print(variabilidade)

# 4.3.3 - Quantificar valores omissos ou indeterminados

nd_count = (ptd["Nível de Utilização [%]"] == "N/D").sum()
lt20_count = ptd["Nível de Utilização [%]"].astype(str).str.contains("<20").sum()

total = len(ptd)

print()
print("4.3.3: Quantidade de valores omissos ou indeterminados:")
print("Valores N/D:", nd_count)
print("Valores <20:", lt20_count)
print("Total de registos:", total)

print("Percentagem N/D:", round((nd_count/total)*100,2), "%")
print("Percentagem <20:", round((lt20_count/total)*100,2), "%")

# remover NaN para análise
utilizacao = ptd["Utilizacao_decimal"].dropna()

plt.figure(figsize=(8,5))
plt.boxplot(utilizacao)

plt.title("Distribuição do Nível de Utilização da Rede")
plt.ylabel("Utilização da Rede")

plt.show()

# 4.3.4 - Estatísticas do nível de utilização para alguns concelhos

concelhos = ["Coimbra", "Évora", "Braga", "Faro"]

dados_concelhos = ptd[ptd["Concelho"].isin(concelhos)]

estatisticas = dados_concelhos.groupby("Concelho")["Utilizacao_decimal"].agg([
    "mean",
    "std",
    "skew",
    pd.Series.kurt
])

# adicionar quartis
estatisticas["Q1"] = dados_concelhos.groupby("Concelho")["Utilizacao_decimal"].quantile(0.25)
estatisticas["Q2"] = dados_concelhos.groupby("Concelho")["Utilizacao_decimal"].quantile(0.50)
estatisticas["Q3"] = dados_concelhos.groupby("Concelho")["Utilizacao_decimal"].quantile(0.75)

# reorganizar colunas
estatisticas = estatisticas[["mean", "Q1", "Q2", "Q3", "std", "skew", "kurt"]]

# arredondar para 4 casas decimais
estatisticas = estatisticas.round(4)

print()
print("4.3.4: Estatísticas do nível de utilização por concelho")
print(estatisticas)

# 4.4 - TESTES DE HIPÓTESES

# 4.4.1 - Teste ao nível médio de ocupação da rede (< 60%)

print("\n--- 4.4.1: Teste ao nível médio de ocupação da rede (< 60%) ---")

# 1. Selecionar uma amostra aleatória de 50 concelhos
amostra_50 = df_final.sample(n=50, random_state=42)
util_amostra = amostra_50["Util_Media"]

print(f"Média da amostra: {util_amostra.mean():.4f}")
print(f"Mediana da amostra: {util_amostra.median():.4f}\n")

# 2. Testar a normalidade dos dados com o teste de Shapiro-Wilk
stat_shapiro, p_shapiro = stats.shapiro(util_amostra)
print("Teste de Ajustamento (Shapiro-Wilk):")
print(f"Estatística: {stat_shapiro:.4f} | p-value: {p_shapiro:.4f}\n")

mu_0 = 0.60
alpha = 0.05

print("--- Decisão do Teste ---")
# 3. Escolher o teste mediante o resultado do Shapiro-Wilk
if p_shapiro > alpha:
    print("Como p-value > 0.05, NÃO se rejeita a normalidade dos dados.")
    print("-> Avança-se com o Teste Paramétrico: t-Student para 1 amostra.\n")

    # Teste t-Student (unilateral à esquerda)
    t_stat, p_t = stats.ttest_1samp(util_amostra, popmean=mu_0, alternative='less')

    print(f"Estatística t: {t_stat:.4f} | p-value do t-teste: {p_t:.4e}")
    if p_t < alpha:
        print("Conclusão: Rejeita-se H0. Existe evidência estatística de que o nível de ocupação é inferior a 60%.")
    else:
        print("Conclusão: Não se rejeita H0. Não existe evidência estatística de que o nível de ocupação é inferior a 60%.")

else:
    print("Como p-value <= 0.05, REJEITA-SE a normalidade dos dados.")
    print("-> Avança-se com o Teste Não Paramétrico: Teste de Wilcoxon.\n")

    # Avaliar assimetria (skewness) para o teste de Wilcoxon
    assimetria = stats.skew(util_amostra)
    print(f"Assimetria (Skewness): {assimetria:.4f}")
    if abs(assimetria) < 0.1:
        print("(Distribuição simétrica - Condição ideal para Wilcoxon cumprida)")
    elif 0.1 <= abs(assimetria) <= 1:
        print("(Distribuição moderadamente assimétrica)")
    else:
        print("(Distribuição fortemente assimétrica)")

    # Teste de Wilcoxon (unilateral à esquerda)
    w_stat, p_w = stats.wilcoxon(util_amostra - mu_0, alternative='less')

    print(f"\nEstatística W: {w_stat:.4f} | p-value do Wilcoxon: {p_w:.4e}")
    if p_w < alpha:
        print("Conclusão: Rejeita-se H0. Existe evidência estatística de que o nível de ocupação é inferior a 60%.")
    else:
        print("Conclusão: Não se rejeita H0. Não existe evidência estatística de que o nível de ocupação é inferior a 60%.")

# 4.4.2 - Teste de diferença entre concelhos Modernizados e Ineficientes

print()
print("\n--- 4.4.2: Comparação entre concelhos Modernizados e Ineficientes ---")

# calcular mediana do rácio de ineficiência
mediana = df_final["Rate_Ineficiencia"].median()

# classificar concelhos
df_final["Grupo"] = np.where(
    df_final["Rate_Ineficiencia"] > mediana,
    "Ineficiente",
    "Modernizado"
)

print("Mediana do rácio de ineficiência:", mediana)
print(df_final["Grupo"].value_counts())

modernizados = df_final[df_final["Grupo"] == "Modernizado"]
ineficientes = df_final[df_final["Grupo"] == "Ineficiente"]

amostra_mod = modernizados.sample(n=30, random_state=42)
amostra_inef = ineficientes.sample(n=30, random_state=42)

util_mod = amostra_mod["Util_Media"]
util_inef = amostra_inef["Util_Media"]

print("\nMédias das amostras:")
print("Modernizados:", util_mod.mean())
print("Ineficientes:", util_inef.mean())

shapiro_mod = shapiro(util_mod)
shapiro_inef = shapiro(util_inef)

print("\nTeste de normalidade (Shapiro-Wilk)")

print("Modernizados p-value:", shapiro_mod.pvalue)
print("Ineficientes p-value:", shapiro_inef.pvalue)

teste = ttest_ind(util_mod, util_inef)

print("\nTeste t para duas amostras independentes")
print("p-value:", teste.pvalue)

# 4.4.3 - ANOVA entre três perfis regionais

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

print("\n--- 4.4.3: ANOVA entre regiões ---")

# Trabalhar sobre a base consolidada
df_anova = df_final.copy()

# Criar distrito a partir de CodDistritoConcelho
df_anova["Distrito"] = (df_anova["CodDistritoConcelho"] // 100).map({
    1: "Aveiro",
    2: "Beja",
    3: "Braga",
    6: "Coimbra",
    7: "Évora",
    11: "Lisboa",
    12: "Portalegre",
    13: "Porto",
    15: "Setúbal"
})

# Criar os 3 grupos pedidos no enunciado
grupo1 = df_anova[df_anova["Distrito"].isin(["Porto", "Braga", "Coimbra"])]
grupo2 = df_anova[df_anova["Distrito"].isin(["Lisboa", "Setúbal", "Aveiro"])]
grupo3 = df_anova[df_anova["Distrito"].isin(["Évora", "Beja", "Portalegre"])]

# Verificar se há pelo menos 25 concelhos por grupo
print("Número de concelhos disponíveis por grupo:")
print("Grupo 1:", len(grupo1))
print("Grupo 2:", len(grupo2))
print("Grupo 3:", len(grupo3))

# Amostragem aleatória de 25 concelhos por grupo
amostra1 = grupo1.sample(n=25, random_state=42)["Util_Media"]
amostra2 = grupo2.sample(n=25, random_state=42)["Util_Media"]
amostra3 = grupo3.sample(n=25, random_state=42)["Util_Media"]

# Médias das amostras
print("\nMédias das amostras:")
print("Grupo 1 - Norte/Centro Litoral:", round(amostra1.mean(), 4))
print("Grupo 2 - Lisboa/Litoral Sul:", round(amostra2.mean(), 4))
print("Grupo 3 - Interior/Alentejo:", round(amostra3.mean(), 4))

# Boxplot para apoio visual
plt.figure(figsize=(9, 6))
plt.boxplot(
    [amostra1, amostra2, amostra3],
    labels=["Norte/Centro", "Lisboa/Litoral", "Interior/Alentejo"]
)
plt.title("Comparação do nível médio de ocupação da rede por região")
plt.ylabel("Util_Media")
plt.show()

# ANOVA
anova = f_oneway(amostra1, amostra2, amostra3)

print("\nResultado da ANOVA:")
print("F-statistic:", anova.statistic)
print("p-value:", anova.pvalue)

alpha = 0.05

if anova.pvalue < alpha:
    print("\nConclusão: rejeita-se H0. Existem diferenças significativas entre pelo menos dois grupos.")

    # Post-hoc de Tukey
    dados_tukey = pd.DataFrame({
        "valor": pd.concat([amostra1, amostra2, amostra3], ignore_index=True),
        "grupo": (["Norte/Centro Litoral"] * len(amostra1) +
                  ["Lisboa/Litoral Sul"] * len(amostra2) +
                  ["Interior/Alentejo"] * len(amostra3))
    })

    tukey = pairwise_tukeyhsd(
        endog=dados_tukey["valor"],
        groups=dados_tukey["grupo"],
        alpha=0.05
    )

    print("\nTeste post-hoc de Tukey:")
    print(tukey)

else:
    print("\nConclusão: não se rejeita H0. Não há evidência estatística de diferenças significativas entre os grupos.")