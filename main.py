import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, pearsonr, ttest_ind
import scipy.stats as stats
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd



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
top_ineficientes = ip_group.sort_values("P_IP_Inef", ascending=False).head(10).copy()

mapa_concelhos = dict(zip(ip['CodDistritoConcelho'], ip['Concelho']))

top_ineficientes["NomeConcelho"] = top_ineficientes["CodDistritoConcelho"].map(mapa_concelhos)

print()
print("Concelhos com maior potência ineficiente:")
print(top_ineficientes[["NomeConcelho", "P_IP_Inef"]])

plt.figure(figsize=(10,6))

plt.bar(top_ineficientes["NomeConcelho"], top_ineficientes["P_IP_Inef"], color='#e74c3c', alpha=0.8)

plt.title("Top 10 Concelhos com Maior Potência de Iluminação Ineficiente", fontsize=14, pad=15)
plt.xlabel("Concelho", fontsize=12)
plt.ylabel("Potência Ineficiente (kW)", fontsize=12)

# ha='right' ajuda a alinhar os nomes compridos (como Sintra ou Vila Nova de Gaia)
plt.xticks(rotation=45, ha='right')

sns.despine()

plt.tight_layout()
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

# filtrar apenas os 4 distritos
distritos = ["Coimbra", "Évora", "Braga", "Faro"]

# usar dados agregados por concelho
ptd_group["Distrito_cod"] = ptd_group["CodDistritoConcelho"] // 100

map_distritos = {
    3: "Braga",
    6: "Coimbra",
    7: "Évora",
    8: "Faro"
}

ptd_group["Distrito"] = ptd_group["Distrito_cod"].map(map_distritos)

dados_concelhos = ptd_group[ptd_group["Distrito"].isin(distritos)]

estatisticas = dados_concelhos.groupby("Distrito")["Util_Media"].agg([
    "mean",
    "std",
    "skew",
    pd.Series.kurt
])

estatisticas["Q1"] = dados_concelhos.groupby("Distrito")["Util_Media"].quantile(0.25)
estatisticas["Q2"] = dados_concelhos.groupby("Distrito")["Util_Media"].quantile(0.50)
estatisticas["Q3"] = dados_concelhos.groupby("Distrito")["Util_Media"].quantile(0.75)

# organizar
estatisticas = estatisticas[["mean", "Q1", "Q2", "Q3", "std", "skew", "kurt"]]

# arredondar
estatisticas = estatisticas.round(4)

print()
print("4.3.4: Estatísticas do nível de utilização por distrito")
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

# Teste de normalidade
shapiro_mod = shapiro(util_mod)
shapiro_inef = shapiro(util_inef)

print("\nTeste de normalidade (Shapiro-Wilk)")
print("Modernizados p-value:", shapiro_mod.pvalue)
print("Ineficientes p-value:", shapiro_inef.pvalue)

alpha = 0.05

print("\n--- Decisão do Teste ---")

# Verificar normalidade em ambos os grupos
if shapiro_mod.pvalue > alpha and shapiro_inef.pvalue > alpha:
    print("Ambos os grupos seguem distribuição normal.")
    print("-> Teste Paramétrico: t-Student para amostras independentes\n")

    teste = ttest_ind(util_mod, util_inef, alternative='two-sided')

    print("Estatística t:", teste.statistic)
    print("p-value:", teste.pvalue)

    if teste.pvalue < alpha:
        print("Conclusão: Rejeita-se H0. Existem diferenças significativas entre os grupos.")
    else:
        print("Conclusão: Não se rejeita H0. Não há evidência de diferenças significativas.")

else:
    print("Pelo menos um dos grupos NÃO segue distribuição normal.")
    print("-> Teste Não Paramétrico: Mann-Whitney\n")

    teste = stats.mannwhitneyu(util_mod, util_inef, alternative='two-sided')

    print("Estatística U:", teste.statistic)
    print("p-value:", teste.pvalue)

    if teste.pvalue < alpha:
        print("Conclusão: Rejeita-se H0. Existem diferenças significativas entre os grupos.")
    else:
        print("Conclusão: Não se rejeita H0. Não há evidência de diferenças significativas.")

# 4.4.3 - ANOVA entre três perfis regionais

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


# 4.4.4 - Correlação entre Capacidade PTD e Iluminação Pública

print("\n--- 4.4.4: Correlação de Pearson (Cap_PTD vs P_IP_Total) ---")

# 1. Isolar as variáveis (garantir que não há NaNs nas duas colunas)
df_corr = df_final[["Cap_PTD", "P_IP_Total"]].dropna()
x = df_corr["Cap_PTD"]
y = df_corr["P_IP_Total"]

# 2. Calcular a correlação de Pearson
coef_corr_pearson, p_value = pearsonr(x, y)

print(f"Coeficiente de Correlação de Pearson (r): {coef_corr_pearson:.4f}")
print(f"Valor de prova do teste (p-value): {p_value:.4e}")

# 3. Processo de decisão
alpha = 0.05
print("\n--- Decisão do Teste ---")
if p_value < alpha:
    print("Conclusão: Rejeita-se H0.")
    print("Existe uma relação linear estatisticamente significativa entre a capacidade de transformação instalada e a carga de iluminação pública.")
else:
    print("Conclusão: Não se rejeita H0.")
    print("Não existe evidência estatística de uma relação linear significativa.")

# 4. Interpretação do Coeficiente (r)
print("\n--- Interpretação do Coeficiente (r) ---")
if coef_corr_pearson > 0.7:
    forca = "forte e positiva"
elif coef_corr_pearson > 0.3:
    forca = "moderada e positiva"
elif coef_corr_pearson > 0:
    forca = "fraca e positiva"
elif coef_corr_pearson < -0.7:
    forca = "forte e negativa"
elif coef_corr_pearson < -0.3:
    forca = "moderada e negativa"
else:
    forca = "fraca e negativa"

print(f"O valor de r ({coef_corr_pearson:.4f}) indica uma correlação {forca}.")
print("Isto significa que concelhos com maior capacidade instalada tendem a ter um maior consumo de iluminação pública.")
# 5. Gráfico de Dispersão (Visualização)
plt.figure(figsize=(8, 5))
sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("Relação entre Capacidade de Transformação (PTD) e Iluminação Pública (IP)")
plt.xlabel("Capacidade Nominal de Transformação - PTD (kVA)")
plt.ylabel("Potência Total Instalada - IP (kW)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# 4.5. CORRELAÇÃO E REGRESSÃO

print("\n--- 4.5: Preparação dos dados para regressão ---")

# criar distrito no df_final
df_final["Distrito"] = (df_final["CodDistritoConcelho"] // 100).map({
    1: "Aveiro",
    3: "Braga",
    11: "Lisboa",
    13: "Porto"
})

# filtrar distritos pretendidos
df_modelo = df_final[df_final["Distrito"].isin(["Aveiro", "Porto", "Lisboa", "Braga"])]

# selecionar variáveis relevantes
df_modelo = df_modelo[[
    "P_IP_Total",
    "Cap_PTD",
    "Rate_Ineficiencia",
    "Util_Media"
]]

# remover NaN
df_modelo = df_modelo.dropna()

print("Dimensão do dataset final:", df_modelo.shape)
print(df_modelo.head())

print("\n--- 4.5: Matriz de Correlação ---")

correlacao = df_modelo.corr()

print(correlacao)

import seaborn as sns

plt.figure(figsize=(6,5))
sns.heatmap(correlacao, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.show()

# 4.5.1 - Modelo de regressão linear múltipla

import statsmodels.api as sm

print("\n--- 4.5.1: Regressão Linear Múltipla ---")

X = df_modelo[["P_IP_Total", "Cap_PTD", "Rate_Ineficiencia"]]
Y = df_modelo["Util_Media"]

# adicionar constante (β0)
X = sm.add_constant(X)

modelo = sm.OLS(Y, X).fit()

print(modelo.summary())

# 4.5.2 - Verificar os resíduos

# Primeiro, verificar a normalidade dos resíduos

residuos = modelo.resid

shapiro_test = stats.shapiro(residuos)

print("\nTeste de normalidade dos resíduos (Shapiro):")
print("p-value:", shapiro_test.pvalue)

stats.probplot(residuos, dist="norm", plot=plt)
plt.title("Q-Q Plot dos Resíduos")
plt.show()

# Segundo, verificar a independência dos resíduos

from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(modelo.resid)

print("\nTeste de Durbin-Watson:")
print("Estatística DW:", round(dw, 4))

# Terceiro, verificar a homocedasticidade dos resíduos

valores_ajustados = modelo.fittedvalues

plt.scatter(valores_ajustados, residuos)
plt.axhline(y=0)
plt.xlabel("Valores Ajustados")
plt.ylabel("Resíduos")
plt.title("Resíduos vs Valores Ajustados")
plt.show()

# 4.5.3 - Verificar multicolinearidade

from statsmodels.stats.outliers_influence import variance_inflation_factor

print("\n--- 4.5.3: VIF ---")

X_vif = X.copy()

vif_data = pd.DataFrame()
vif_data["Variável"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print(vif_data)

# 1. P_IP_Total vs Util_Media
plt.figure(figsize=(6,5))
sns.regplot(x="P_IP_Total", y="Util_Media", data=df_modelo)
plt.title("P_IP_Total vs Util_Media")
plt.xlabel("Potência IP Total (kW)")
plt.ylabel("Utilização Média")
plt.show()


# 2. Cap_PTD vs Util_Media
plt.figure(figsize=(6,5))
sns.regplot(x="Cap_PTD", y="Util_Media", data=df_modelo)
plt.title("Capacidade PTD vs Util_Media")
plt.xlabel("Capacidade PTD (kVA)")
plt.ylabel("Utilização Média")
plt.show()


# 3. Rate_Ineficiencia vs Util_Media
plt.figure(figsize=(6,5))
sns.regplot(x="Rate_Ineficiencia", y="Util_Media", data=df_modelo)
plt.title("Taxa de Ineficiência vs Util_Media")
plt.xlabel("Taxa de Ineficiência")
plt.ylabel("Utilização Média")
plt.show()


# 5. Previsão para aveiro

mapa_concelhos = dict(zip(ip['CodDistritoConcelho'], ip['Concelho']))
lista_aveiro = [101,102,103,104,105,106,107,108,109]
concelhos_aveiro = df_final[df_final["CodDistritoConcelho"].isin(lista_aveiro)].copy()
concelhos_aveiro["NomeConcelho"] = concelhos_aveiro["CodDistritoConcelho"].map(mapa_concelhos)
X_aveiro = concelhos_aveiro[["P_IP_Total", "Cap_PTD", "Rate_Ineficiencia"]]
X_aveiro = sm.add_constant(X_aveiro)
concelhos_aveiro["Previsao_Util"] = modelo.predict(X_aveiro)

# resultado final
print("\n--- 4.5.5 ---")
print(concelhos_aveiro[[
    "NomeConcelho",
    "Util_Media",
    "Previsao_Util"
]])

# gráfigo
plt.figure(figsize=(10, 6))
df_plot = concelhos_aveiro.melt(id_vars="NomeConcelho", value_vars=["Util_Media", "Previsao_Util"])
sns.barplot(data=df_plot, x="NomeConcelho", y="value", hue="variable", palette="viridis")
plt.title("Aveiro: Utilização Real vs. Prevista por Concelho")
plt.ylabel("Nível de Utilização")
plt.xticks(rotation=45, ha='right') # ha='right' alinha bem os nomes inclinados
plt.legend(title="Legenda")
plt.tight_layout()
plt.show()



# 6. REDUÇÃO ESPERADA (β3)

beta3 = modelo.params["Rate_Ineficiencia"]
delta = -0.20
impacto = beta3 * delta

print("\n--- 4.5.6 ---")
print("Beta3:", beta3)
print("Redução esperada no nível de ocupação:", impacto)




# 7. INTERVALOS DE CONFIANÇA


df_previsao = df_final[df_final["Distrito"].isin(["Aveiro", "Porto", "Lisboa", "Braga"])].copy()
df_previsao = df_previsao[[
    "CodDistritoConcelho",
    "P_IP_Total",
    "Cap_PTD",
    "Rate_Ineficiencia",
    "Util_Media"
]].dropna()

X_all = df_previsao[["P_IP_Total", "Cap_PTD", "Rate_Ineficiencia"]]
X_all = sm.add_constant(X_all)
pred = modelo.get_prediction(X_all)
pred_summary = pred.summary_frame(alpha=0.05)

# adicionar resultados
df_previsao["Pred"] = pred_summary["mean"]
df_previsao["IC_inf"] = pred_summary["mean_ci_lower"]
df_previsao["IC_sup"] = pred_summary["mean_ci_upper"]

concelhos_viaveis = df_previsao.sort_values("Pred")

print("\n--- 4.5.7 ---")
print(concelhos_viaveis[[
    "CodDistritoConcelho",
    "Pred",
    "IC_inf",
    "IC_sup"
]].head(10))  # top 10 mais viáveis

# gráfico
top10 = concelhos_viaveis.head(10).copy()
top10 = top10.sort_values("Pred", ascending=False)

mapa_concelhos = dict(zip(ip['CodDistritoConcelho'], ip['Concelho']))
top10["NomeConcelho"] = top10["CodDistritoConcelho"].map(mapa_concelhos)

erro_inf = top10["Pred"] - top10["IC_inf"]
erro_sup = top10["IC_sup"] - top10["Pred"]

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

limite_esq = top10["IC_inf"].min() - 0.02
limite_dir = top10["IC_sup"].max() + 0.02
plt.xlim(limite_esq, limite_dir)

barras = plt.barh(
    top10["NomeConcelho"], 
    top10["Pred"], 
    color='#4C72B0',      
    edgecolor='none',     
    alpha=0.9,            
    height=0.55,
    label='Previsão Média (Utilização)'
)

plt.errorbar(
    top10["Pred"], 
    top10["NomeConcelho"], 
    xerr=[erro_inf, erro_sup], 
    fmt='none',           
    ecolor='#333333',     
    elinewidth=2,         
    capsize=0,            
    zorder=3,
    label='Intervalo de Confiança (95%)'
)

for i, barra in enumerate(barras):
    largura = barra.get_width()
    plt.text(limite_esq + 0.003, barra.get_y() + barra.get_height()/2, 
             f'{largura:.4f}', 
             va='center', color='white', fontweight='bold', fontsize=10)

plt.title("Top 10 Concelhos Mais Viáveis para Instalação de VE\n(Menor Ocupação de Rede Prevista com IC 95%)", fontsize=14, pad=15)
plt.xlabel("Nível Médio de Utilização Previsto (Decimal)", fontsize=12)
plt.ylabel("Concelho", fontsize=12)

plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

sns.despine(left=True, bottom=False)

plt.tight_layout()
plt.show()