import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree

# ============================================================
# Secção 4.1 - Análise Exploratória de Dados
# Ponto 1: Carregamento, dimensão e sumário dos dados
# ============================================================

# ------------------------------------------------------------
# 1. Carregar o ficheiro
# ------------------------------------------------------------
df = pd.read_excel("PTD_level_dataset.xlsx")

# ------------------------------------------------------------
# 2. Dimensão do dataset
# ------------------------------------------------------------
print("=" * 60)
print("DIMENSÃO DO DATASET")
print("=" * 60)
print(f"  Número de registos (linhas) : {df.shape[0]}")
print(f"  Número de variáveis (colunas): {df.shape[1]}")
print()

# ------------------------------------------------------------
# 3. Sumário estatístico das variáveis numéricas
#    (média, mediana, min, max, desvio padrão, quartis)
# ------------------------------------------------------------
print("=" * 60)
print("SUMÁRIO ESTATÍSTICO (variáveis numéricas)")
print("=" * 60)

summary = df.describe().T
summary["median"] = df.median(numeric_only=True)
summary = summary[["count", "mean", "median", "std", "min", "25%", "75%", "max"]]
summary.columns = ["Count", "Média", "Mediana", "Desvio Padrão", "Mín", "Q1 (25%)", "Q3 (75%)", "Máx"]

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
print(summary)
print()

# ------------------------------------------------------------
# 4. Valores omissos
# ------------------------------------------------------------
print("=" * 60)
print("VALORES OMISSOS")
print("=" * 60)

missing = pd.DataFrame({
    "Nº omissos": df.isnull().sum(),
    "% omissos": (df.isnull().sum() / len(df) * 100).round(2)
})
missing = missing[missing["Nº omissos"] > 0].sort_values("% omissos", ascending=False)

if missing.empty:
    print("  Não existem valores omissos no dataset.")
else:
    print(missing.to_string())
print()

# ------------------------------------------------------------
# 5. Variáveis numéricas
# ------------------------------------------------------------
print("=" * 60)
print("RESUMO DE OUTLIERS (método IQR)")
print("=" * 60)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_info = []
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    pct_out = round(n_out / len(df) * 100, 2)
    outlier_info.append({"Variável": col, "Nº outliers": n_out, "% outliers": pct_out})

outlier_df = pd.DataFrame(outlier_info).sort_values("% outliers", ascending=False)
print(outlier_df[outlier_df["Nº outliers"] > 0].to_string(index=False))
print()

# ============================================================
# Secção 4.1 - Análise Exploratória de Dados
# Ponto2: Exploração Visual dos Dados
# ============================================================


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Definir o estilo visual e paletes padrão dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

# ============================================================
# 1. Variáveis Numéricas (Histogramas e Boxplots)
# ============================================================

# Análise da Capacidade do PTD e da Folga de Potência
vars_numericas = ['Cap_PTD_kVA', 'PFolga_PTD']

for col in vars_numericas:
    if col in df.columns:
        plt.figure(figsize=(12, 4))
        # Histograma
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True, color='skyblue', bins=40)
        plt.title(f'Histograma: Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col].dropna(), color='skyblue')
        plt.title(f'Boxplot: Deteção de Outliers em {col}')
        plt.xlabel(col)

        plt.tight_layout()
        plt.show()

# ============================================================
# 2. Variáveis Categóricas (Gráfico de Barras / Contagem)
# ============================================================
if 'Tipo Construtivo' in df.columns:
    plt.figure(figsize=(12, 6))

    # 1. Obter a contagem ordenada dos dados
    tipo_counts = df['Tipo Construtivo'].value_counts()
    order = tipo_counts.index

    # 2. Criar o gráfico base SEM usar o argumento 'hue' para garantir as barras largas
    ax = sns.countplot(
        data=df,
        y='Tipo Construtivo',
        order=order,
        color='skyblue'
    )

    # 3. Aplicar manualmente as cores da palete 'viridis' a cada barra
    num_barras = len(order)
    cores_palete = sns.color_palette('viridis', n_colors=num_barras)

    for i, patch in enumerate(ax.patches):
        if i < num_barras:
            patch.set_facecolor(cores_palete[i])

    # 4. ADICIONAR OS VALORES REAIS À FRENTE DE CADA BARRA
    ax.bar_label(
        ax.containers[0],
        fmt='%d',
        padding=8,
        fontsize=10.5,
        weight='semibold',
        color='#2c3e50'
    )

    ax.set_title('Frequência dos PTDs por Tipo Construtivo', fontsize=13, pad=15, weight='bold')
    ax.set_xlabel('Contagem de Registos', fontsize=11, labelpad=10)
    ax.set_ylabel('Tipo Construtivo', fontsize=11, labelpad=10)

    ax.set_xlim(0, tipo_counts.max() * 1.15)

    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()

# ============================================================
# 3. Matriz de Correlação (Heatmap)
# ============================================================
# Selecionar as variáveis numéricas mais relevantes do problema para não sobrecarregar o heatmap
colunas_interesse = ['Cap_PTD_kVA', 'PFolga_PTD', 'Util_Decimal', 'Pot_Contratada_kVA', 'N_Clientes', 'PVE_PTD']
colunas_validas = [c for c in colunas_interesse if c in df.columns]

plt.figure(figsize=(8, 6))
matriz_corr = df[colunas_validas].corr(method='pearson')

# Desenhar o Heatmap
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriz de Correlação de Pearson (Heatmap)')
plt.tight_layout()
plt.show()

# ============================================================
# 4. Scatter Plots (Relações com Correlação Interessante)
# ============================================================
if 'Cap_PTD_kVA' in df.columns and 'PFolga_PTD' in df.columns:
    plt.figure(figsize=(7, 5))
    # Amostra de 2000 pontos aleatórios para evitar sobreposição visual
    df_sample = df.dropna(subset=['Cap_PTD_kVA', 'PFolga_PTD']).sample(min(2000, len(df)), random_state=42)

    sns.scatterplot(data=df_sample, x='Cap_PTD_kVA', y='PFolga_PTD', alpha=0.6, color='teal')
    plt.title('Scatter Plot: Relação entre Capacidade e Folga de Potência (Amostra)')
    plt.xlabel('Capacidade do PTD (Cap_PTD_kVA)')
    plt.ylabel('Folga de Potência (PFolga_PTD)')
    plt.tight_layout()
    plt.show()

# ============================================================
# Secção 4.1.3 - Pré-processamento dos Dados
# ============================================================

import pandas as pd
import numpy as np

df_processed = df.copy()

print("=" * 60)
print("TRATAMENTO DE VALORES OMISSOS")
print("=" * 60)

missing_values = df_processed.isnull().sum()
missing_values = missing_values[missing_values > 0]

if missing_values.empty:
    print("Não existem valores omissos no dataset.")
else:
    print(missing_values)

# ------------------------------------------------------------
# Remover linhas com omissos nas variáveis-alvo
# ------------------------------------------------------------

antes = len(df_processed)

df_processed = df_processed.dropna(
    subset=[
        "PFolga_PTD",
        "Util_Decimal"
    ]
)

depois = len(df_processed)

print(f"\nLinhas removidas por omissos nas variáveis-alvo: {antes - depois}")
print(f"Registos após remoção: {depois}")

# ------------------------------------------------------------
# Imputar restantes variáveis numéricas
# ------------------------------------------------------------

num_cols = df_processed.select_dtypes(
    include=np.number
).columns

for col in num_cols:
    if df_processed[col].isnull().sum() > 0:
        df_processed[col] = df_processed[col].fillna(
            df_processed[col].median()
        )

# ------------------------------------------------------------
# Imputar variáveis categóricas
# ------------------------------------------------------------

cat_cols = df_processed.select_dtypes(
    include='object'
).columns

for col in cat_cols:
    if df_processed[col].isnull().sum() > 0:
        df_processed[col] = df_processed[col].fillna(
            df_processed[col].mode()[0]
        )

print("\nValores omissos após tratamento:")
print(df_processed.isnull().sum().sum())

# ============================================================
# 2. Seleção de Variáveis Relevantes
# ============================================================

print("\n" + "=" * 60)
print("SELEÇÃO DE VARIÁVEIS")
print("=" * 60)

colunas_remover = [
    "Código de Instalação",
    "Coordenadas Geográficas",
    "Potência instalada [kVA]",
    "D_PTD",
    "D_PTD_LED",
    "Concelho",
    "Nível de Utilização [%]"
]

colunas_existentes = [
    c for c in colunas_remover
    if c in df_processed.columns
]

df_processed.drop(
    columns=colunas_existentes,
    inplace=True
)

print("Variáveis removidas:")
print(colunas_existentes)

# ============================================================
# 3. Transformação de Variáveis Categóricas
# ============================================================

print("\n" + "=" * 60)
print("TRANSFORMAÇÃO DE VARIÁVEIS CATEGÓRICAS")
print("=" * 60)

cat_cols = df_processed.select_dtypes(
    include='object'
).columns.tolist()

print("Variáveis categóricas encontradas:")
print(cat_cols)

df_processed = pd.get_dummies(
    df_processed,
    columns=cat_cols,
    drop_first=True
)

print("\nTransformação concluída.")

# Converter bool para int
bool_cols = df_processed.select_dtypes(
    include="bool"
).columns

df_processed[bool_cols] = df_processed[bool_cols].astype(int)

# ============================================================
# 4. Standardização
# ============================================================

print("\n" + "=" * 60)
print("NORMALIZAÇÃO / STANDARDIZAÇÃO")
print("=" * 60)

print(
    "A standardização NÃO foi aplicada nesta fase.\n"
    "Será realizada dentro dos folds dos modelos "
    "para evitar data leakage."
)

# ============================================================
# 5. Dataset Final
# ============================================================

print("\n" + "=" * 60)
print("DATASET FINAL PRONTO PARA MODELAGEM")
print("=" * 60)

print(f"Dimensão final: {df_processed.shape}")
print(df_processed.dtypes.value_counts())

print("\nNúmero total de variáveis:")
print(df_processed.shape[1])

# ============================================================
# Secção 4.2 - Regressão
# ============================================================

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Ponto 4.2.1: Diagrama de correlação para PFolga_PTD
# ------------------------------------------------------------
print("=" * 60)
print("4.2.1: DIAGRAMA DE CORRELAÇÃO")
print("=" * 60)

# Calculamos a correlação de Pearson de todas as variáveis com a nossa variável alvo
# Nota: Usamos df_processed ou df após tratamento de nulos/categóricas para garantir valores numéricos
corr_matrix = df.corr(method='pearson', numeric_only=True)
target_corr = corr_matrix['PFolga_PTD'].sort_values(ascending=False)

print("Correlação de Pearson das variáveis com 'PFolga_PTD':")
print(target_corr)
print()

# Visualização do gráfico de barras das correlações
plt.figure(figsize=(10, 6))
sns.barplot(x=target_corr.values, y=target_corr.index, hue=target_corr.index, palette='coolwarm', legend=False)
plt.title('Correlação das Variáveis com a Folga de Potência (PFolga_PTD)')
plt.xlabel('Coeficiente de Correlação de Pearson')
plt.ylabel('Variáveis')
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Ponto 4.2.2: Regressão Linear Simples com K-Fold Cross Validation
# ------------------------------------------------------------
print("=" * 60)
print("4.2.2: REGRESSÃO LINEAR SIMPLES (K-FOLD)")
print("=" * 60)

# Escolha da variável explicativa (X): 
# Com base na correlação e na lógica do problema, escolhemos 'Cap_PTD_kVA' 
# porque a capacidade total do posto é o fator estrutural que mais dita a folga disponível.
variavel_explicativa = 'Cap_PTD_kVA'
print(f"-> Variável explicativa relevante selecionada: '{variavel_explicativa}'\n")

X_simple = df_processed[[variavel_explicativa]].values
y_simple = df_processed["PFolga_PTD"].values

# Configuração do K-Fold Cross Validation (por exemplo, k=5 ou k=10, robusto e computacionalmente eficiente)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Listas para armazenar as métricas de cada fold
fold_mae = []
fold_rmse = []

# Listas para guardar os coeficientes obtidos em cada fold para depois calcular a média da função linear
coefficients = []
intercepts = []

# Loop manual do K-Fold para evitar data leakage na Standardização
for fold, (train_index, test_index) in enumerate(kf.split(X_simple), 1):
    # Divisão dos dados em treino e teste para este fold
    X_train, X_test = X_simple[train_index], X_simple[test_index]
    y_train, y_test = y_simple[train_index], y_simple[test_index]

    # Aplicar o StandardScaler dentro do fold (Treinado apenas no Treino, aplicado no Teste)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Inicializar e treinar o modelo de Regressão Linear Simples
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Fazer previsões no conjunto de teste/validação do fold
    y_pred = lr_model.predict(X_test_scaled)

    # Calcular as métricas do fold
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    fold_mae.append(mae)
    fold_rmse.append(rmse)

    coefficients.append(lr_model.coef_[0])
    intercepts.append(lr_model.intercept_)

    print(f"  Fold {fold} -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")

# Calcular a média final das métricas de performance
final_mae = np.mean(fold_mae)
final_rmse = np.mean(fold_rmse)

# Calcular a média dos parâmetros para apresentar a função linear final representativa
final_coef = np.mean(coefficients)
final_intercept = np.mean(intercepts)

print("-" * 60)
print(f"MÉDIA FINAL VALIDAÇÃO CRUZADA ({k}-folds):")
print(f"  Média MAE  : {final_mae:.4f}")
print(f"  Média RMSE : {final_rmse:.4f}")
print("-" * 60)

# a) Apresentação da função linear resultante
print("\na) FUNÇÃO LINEAR RESULTANTE (Dados Standardizados):")
print(f"   $$PFolga\_PTD = {final_intercept:.4f} + ({final_coef:.4f} \\times {variavel_explicativa}\_scaled)$$")
print()

# b) Visualização da reta correspondente ao modelo e o diagrama de dispersão
print("b) Gráfico de dispersão com a reta de regressão.")

# Para a visualização, ajustamos um modelo representativo global escalado para traçar a reta correta
scaler_vis = StandardScaler()
X_scaled_vis = scaler_vis.fit_transform(X_simple)

plt.figure(figsize=(8, 5))
# Desenha uma amostra aleatória de pontos para o gráfico não ficar pesado (conforme fizeste na AED)
sample_indices = np.random.choice(len(X_scaled_vis), min(2000, len(X_scaled_vis)), replace=False)
plt.scatter(X_scaled_vis[sample_indices], y_simple[sample_indices], alpha=0.5, color='teal',
            label='Dados Observados (Amostra)')

# Linha de regressão baseada nos coeficientes médios obtidos via K-Fold
X_line = np.linspace(X_scaled_vis.min(), X_scaled_vis.max(), 100)
y_line = final_intercept + final_coef * X_line
plt.plot(X_line, y_line, color='red', linewidth=3, label='Reta de Regressão (K-Fold)')

plt.title(f'Regressão Linear Simples: PFolga_PTD vs {variavel_explicativa}')
plt.xlabel(f'{variavel_explicativa} (Standardized)')
plt.ylabel('PFolga_PTD (Valor Real)')
plt.legend()
plt.tight_layout()
plt.show()

# c) Exibição explícita dos erros pedidos
print("\nc) MÉTRICAS DE ERRO QUANTIFICADAS:")
print(f"   * Mean Absolute Error (MAE)                 : {final_mae:.4f}")
print(f"   * Root Mean Squared Error (RMSE)            : {final_rmse:.4f}")
print("=" * 60)

# ============================================================
# 4.2.3.a REGRESSÃO LINEAR MÚLTIPLA
# ============================================================

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("=" * 60)
print("4.2.3.a REGRESSÃO LINEAR MÚLTIPLA")
print("=" * 60)

# ------------------------------------------------------------
# Variáveis independentes e variável alvo
# ------------------------------------------------------------

X = df_processed.drop(columns=["PFolga_PTD"])

y = df_processed["PFolga_PTD"]

# ------------------------------------------------------------
# K-Fold
# ------------------------------------------------------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

mae_scores = []
rmse_scores = []

coeficientes = []

# ------------------------------------------------------------
# Validação cruzada
# ------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Standardização dentro do fold
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    mae_scores.append(mae)
    rmse_scores.append(rmse)

    coeficientes.append(model.coef_)

    print(
        f"Fold {fold} -> "
        f"MAE = {mae:.4f} | "
        f"RMSE = {rmse:.4f}"
    )

# ------------------------------------------------------------
# Resultados médios
# ------------------------------------------------------------

print("-" * 60)

print(
    f"MAE Médio : {np.mean(mae_scores):.4f}"
)

print(
    f"RMSE Médio: {np.mean(rmse_scores):.4f}"
)

print("-" * 60)

# ------------------------------------------------------------
# Importância média das variáveis
# ------------------------------------------------------------

coef_medio = np.mean(
    coeficientes,
    axis=0
)

coef_df = pd.DataFrame({
    "Variavel": X.columns,
    "Coeficiente": coef_medio
})

coef_df["AbsCoef"] = abs(
    coef_df["Coeficiente"]
)

coef_df = coef_df.sort_values(
    by="AbsCoef",
    ascending=False
)

print("\nTOP 10 VARIÁVEIS MAIS RELEVANTES")

print(
    coef_df[
        ["Variavel", "Coeficiente"]
    ].head(10)
)

# ------------------------------------------------------------
# Gráfico
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.barplot(
    data=coef_df.head(10),
    x="Coeficiente",
    y="Variavel"
)

plt.title(
    "Top 10 Variáveis Mais Relevantes"
)

plt.tight_layout()
plt.show()

# ============================================================
# 4.2.3.b ÁRVORE DE REGRESSÃO
# ============================================================

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

print("=" * 60)
print("4.2.3.b ÁRVORE DE REGRESSÃO")
print("=" * 60)

# ------------------------------------------------------------
# Variáveis independentes
# ------------------------------------------------------------

X = df_processed.drop(
    columns=["PFolga_PTD"]
)

y = df_processed["PFolga_PTD"]

# ------------------------------------------------------------
# K-Fold Cross Validation
# ------------------------------------------------------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

mae_scores = []
rmse_scores = []

rmse_tree_folds = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    tree = DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )

    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    mae_scores.append(mae)
    rmse_scores.append(rmse)

    rmse_tree_folds.append(rmse)

    print(
        f"Fold {fold} -> "
        f"MAE = {mae:.4f} | "
        f"RMSE = {rmse:.4f}"
    )

# ------------------------------------------------------------
# Resultados médios
# ------------------------------------------------------------

print("-" * 60)

print(
    f"MAE Médio : {np.mean(mae_scores):.4f}"
)

print(
    f"RMSE Médio: {np.mean(rmse_scores):.4f}"
)

print("-" * 60)

# ------------------------------------------------------------
# Modelo final para visualização
# ------------------------------------------------------------

tree_final = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

tree_final.fit(X, y)

# ------------------------------------------------------------
# Visualização da árvore
# ------------------------------------------------------------

plt.figure(figsize=(24,12))

plot_tree(
    tree_final,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Árvore de Regressão para PFolga_PTD"
)

plt.show()

# ------------------------------------------------------------
# Importância das variáveis
# ------------------------------------------------------------

importance = pd.DataFrame({
    "Variavel": X.columns,
    "Importancia": tree_final.feature_importances_
})

importance = importance.sort_values(
    by="Importancia",
    ascending=False
)

print("\nTOP 10 VARIÁVEIS MAIS IMPORTANTES")

print(
    importance.head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importancia",
    y="Variavel"
)

plt.title(
    "Importância das Variáveis"
)

plt.tight_layout()
plt.show()

# ============================================================
# 4.2.3.c SVM (SVR)
# ============================================================

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("4.2.3.c SVM (SVR)")
print("=" * 60)

# ------------------------------------------------------------
# Variáveis independentes e alvo
# ------------------------------------------------------------

X = df_processed.drop(
    columns=["PFolga_PTD"]
)

y = df_processed["PFolga_PTD"]

# ------------------------------------------------------------
# Amostra para reduzir tempo de treino
# ------------------------------------------------------------

#
#X_sample = X.sample(
#    n=10000,
#    random_state=42
#)

#y_sample = y.loc[X_sample.index]

#X = X_sample
#y = y_sample

# ------------------------------------------------------------
# Kernels a comparar
# ------------------------------------------------------------

kernels = [
    "linear",
    "rbf",
    "poly"
]

resultados_svm = []

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ------------------------------------------------------------
# Teste dos kernels
# ------------------------------------------------------------

for kernel in kernels:

    print(f"\nKernel: {kernel}")

    mae_scores = []
    rmse_scores = []

    for train_idx, test_idx in kf.split(X):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Standardização dentro do fold

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        model = SVR(
            kernel=kernel,
            C=1.0,
            epsilon=0.1
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_test_scaled
        )

        mae_scores.append(
            mean_absolute_error(
                y_test,
                y_pred
            )
        )

        rmse_scores.append(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    y_pred
                )
            )
        )

    mae_medio = np.mean(mae_scores)
    rmse_medio = np.mean(rmse_scores)

    resultados_svm.append(
        [kernel, mae_medio, rmse_medio]
    )

    print(
        f"MAE Médio : {mae_medio:.4f}"
    )

    print(
        f"RMSE Médio: {rmse_medio:.4f}"
    )

# ------------------------------------------------------------
# Resumo final
# ------------------------------------------------------------

print("\nResumo dos Kernels")

resultados_svm = pd.DataFrame(
    resultados_svm,
    columns=[
        "Kernel",
        "MAE",
        "RMSE"
    ]
)

print(resultados_svm)

melhor_kernel = resultados_svm.loc[
    resultados_svm["RMSE"].idxmin()
]

print("\nMelhor Kernel:")

print(melhor_kernel)

# ============================================================
# 4.2.3.d REDE NEURONAL PARA REGRESSÃO - MLPRegressor
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor

print("=" * 60)
print("4.2.3.d REDE NEURONAL - MLPRegressor")
print("=" * 60)

# ------------------------------------------------------------
# Variáveis independentes e alvo
# ------------------------------------------------------------

X = df_processed.drop(columns=["PFolga_PTD"]).astype(float)
y = df_processed["PFolga_PTD"]

# ------------------------------------------------------------
# Configurações a testar
# ------------------------------------------------------------

configs = [
    {
        "nome": "Config 1 - Rede simples",
        "hidden_layer_sizes": (32,),
        "alpha": 0.0001,
        "learning_rate_init": 0.001
    },
    {
        "nome": "Config 2 - Rede intermédia",
        "hidden_layer_sizes": (64, 32),
        "alpha": 0.001,
        "learning_rate_init": 0.001
    },
    {
        "nome": "Config 3 - Rede profunda com regularização",
        "hidden_layer_sizes": (128, 64, 32),
        "alpha": 0.01,
        "learning_rate_init": 0.0005
    }
]

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

resultados_nn = []
historicos_loss = {}

# ------------------------------------------------------------
# Treino e avaliação das configurações
# ------------------------------------------------------------

rmse_nn_best = None

for config in configs:

    print("\n" + "=" * 60)
    print(config["nome"])
    print("=" * 60)

    mae_scores = []
    rmse_scores = []

    rmse_folds_config = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Standardização dentro do fold
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPRegressor(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=config["alpha"],  # regularização L2
            learning_rate_init=config["learning_rate_init"],
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        rmse_folds_config.append(rmse)

        print(
            f"Fold {fold} -> "
            f"MAE = {mae:.4f} | "
            f"RMSE = {rmse:.4f}"
        )

        # Guardar loss do primeiro fold para gráfico
        if fold == 1:
            historicos_loss[config["nome"]] = model.loss_curve_

    mae_medio = np.mean(mae_scores)
    rmse_medio = np.mean(rmse_scores)

    if config["nome"] == "Config 2 - Rede intermédia":
        rmse_nn_best = rmse_folds_config.copy()

    resultados_nn.append([
        config["nome"],
        config["hidden_layer_sizes"],
        config["alpha"],
        config["learning_rate_init"],
        mae_medio,
        rmse_medio
    ])

    print("-" * 60)
    print(f"MAE Médio : {mae_medio:.4f}")
    print(f"RMSE Médio: {rmse_medio:.4f}")

# ------------------------------------------------------------
# Resumo final
# ------------------------------------------------------------

resultados_nn = pd.DataFrame(
    resultados_nn,
    columns=[
        "Configuração",
        "Camadas",
        "Alpha L2",
        "Learning Rate",
        "MAE",
        "RMSE"
    ]
)

print("\nResumo das Redes Neuronais:")
print(resultados_nn)

melhor_nn = resultados_nn.loc[
    resultados_nn["RMSE"].idxmin()
]

print("\nMelhor configuração:")
print(melhor_nn)

# ------------------------------------------------------------
# Curvas de loss
# ------------------------------------------------------------


for nome_config, loss_curve in historicos_loss.items():

    plt.figure(figsize=(8, 5))

    plt.plot(
        loss_curve,
        label="Loss Treino"
    )

    plt.title(f"Curva de Loss - {nome_config}")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 4.2.5 CURVAS DE APRENDIZAGEM
# ============================================================

from sklearn.model_selection import learning_curve

tree_model = DecisionTreeRegressor(
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

train_sizes, train_scores, val_scores = learning_curve(
    tree_model,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_rmse = -train_scores.mean(axis=1)
val_rmse = -val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))

plt.plot(
    train_sizes,
    train_rmse,
    marker="o",
    label="Training RMSE"
)

plt.plot(
    train_sizes,
    val_rmse,
    marker="o",
    label="Validation RMSE"
)

plt.title("Learning Curve - Árvore de Regressão")
plt.xlabel("Número de exemplos de treino")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)

plt.show()

from sklearn.neural_network import MLPRegressor

nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42
)

train_sizes, train_scores, val_scores = learning_curve(
    nn_model,
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_rmse = -train_scores.mean(axis=1)
val_rmse = -val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))

plt.plot(
    train_sizes,
    train_rmse,
    marker="o",
    label="Training RMSE"
)

plt.plot(
    train_sizes,
    val_rmse,
    marker="o",
    label="Validation RMSE"
)

plt.title("Learning Curve - Rede Neuronal")
plt.xlabel("Número de exemplos de treino")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)

print(train_rmse)
print(val_rmse)

plt.show()

# ============================================================
# 4.2.6 TESTES ESTATÍSTICOS
# ============================================================

from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

print("=" * 60)
print("4.2.6 TESTES ESTATÍSTICOS")
print("=" * 60)

# Teste t emparelhado

t_stat, p_value = ttest_rel(
    rmse_tree_folds,
    rmse_nn_best
)

print("\nTeste t emparelhado")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")

# Wilcoxon

w_stat, p_wilcoxon = wilcoxon(
    rmse_tree_folds,
    rmse_nn_best
)

print("\nTeste de Wilcoxon")
print(f"statistic = {w_stat:.4f}")
print(f"p-value = {p_wilcoxon:.4f}")

if p_value < 0.05:
    print("\nDiferença estatisticamente significativa (5%).")
else:
    print("\nDiferença NÃO estatisticamente significativa (5%).")


# ============================================================
# 4.3 CLASSIFICAÇÃO
# Criação da variável alvo utilizRede
# ============================================================

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

print("=" * 60)
print("4.3 CLASSIFICAÇÃO - CRIAÇÃO DO ALVO utilizRede")
print("=" * 60)

df_class = df_processed.copy()

# Criar variável categórica com base em Util_Decimal
# Baixo: até 39%
# Médio: 40% a 79%
# Alto: 80% ou superior

df_class["utilizRede"] = pd.cut(
    df_class["Util_Decimal"],
    bins=[-np.inf, 0.39, 0.79, np.inf],
    labels=["baixo", "médio", "alto"]
)

print("Distribuição da variável utilizRede:")
print(df_class["utilizRede"].value_counts())
print()

plt.figure(figsize=(6,4))
sns.countplot(
    data=df_class,
    x="utilizRede",
    order=["baixo", "médio", "alto"]
)
plt.title("Distribuição da variável utilizRede")
plt.xlabel("Nível de utilização da rede")
plt.ylabel("Número de registos")
plt.tight_layout()
plt.show()

# ============================================================
# 4.3.1.a ÁRVORE DE DECISÃO - CLASSIFICAÇÃO
# ============================================================

print("=" * 60)
print("4.3.1.a ÁRVORE DE DECISÃO")
print("=" * 60)

# ------------------------------------------------------------
# Variáveis independentes e alvo
# ------------------------------------------------------------

X = df_class.drop(
    columns=[
        "utilizRede",
        "Util_Decimal",
        "PFolga_PTD"
    ]
)

y = df_class["utilizRede"]

# Converter variáveis booleanas para 0/1
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

# ------------------------------------------------------------
# Stratified K-Fold
# ------------------------------------------------------------

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# guardar previsões finais para relatório
y_true_all = []
y_pred_all = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    tree_clf = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=50,
        min_samples_leaf=25,
        class_weight="balanced",
        random_state=42
    )

    tree_clf.fit(X_train, y_train)

    y_pred = tree_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    accuracy_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

    print(
        f"Fold {fold} -> "
        f"Accuracy = {acc:.4f} | "
        f"Precision = {precision:.4f} | "
        f"Recall = {recall:.4f} | "
        f"F1-score = {f1:.4f}"
    )

print("-" * 60)
print(f"Accuracy Média : {np.mean(accuracy_scores):.4f}")
print(f"Precision Média: {np.mean(precision_scores):.4f}")
print(f"Recall Médio   : {np.mean(recall_scores):.4f}")
print(f"F1-score Médio : {np.mean(f1_scores):.4f}")
print("-" * 60)

# ============================================================
# RELATÓRIO DE CLASSIFICAÇÃO
# ============================================================

print("\nRelatório de Classificação:")
print(
    classification_report(
        y_true_all,
        y_pred_all,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_true_all,
    y_pred_all,
    labels=["baixo", "médio", "alto"]
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["baixo", "médio", "alto"],
    yticklabels=["baixo", "médio", "alto"]
)

plt.title("Matriz de Confusão - Árvore de Decisão")
plt.xlabel("Classe prevista")
plt.ylabel("Classe real")
plt.tight_layout()
plt.show()

# ============================================================
# ÁRVORE FINAL E IMPORTÂNCIA DAS FEATURES
# ============================================================

tree_final = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=50,
    min_samples_leaf=25,
    class_weight="balanced",
    random_state=42
)

tree_final.fit(X, y)

plt.figure(figsize=(26, 12))

plot_tree(
    tree_final,
    feature_names=X.columns,
    class_names=["alto", "baixo", "médio"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Árvore de Decisão para Classificação de utilizRede")
plt.show()

importance = pd.DataFrame({
    "Variavel": X.columns,
    "Importancia": tree_final.feature_importances_
})

importance = importance.sort_values(
    by="Importancia",
    ascending=False
)

print("\nTOP 10 FEATURES MAIS IMPORTANTES")
print(importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importancia",
    y="Variavel"
)

plt.title("Top 10 Features Mais Importantes - Árvore de Decisão")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.tight_layout()
plt.show()

# ============================================================
# 4.3.1.b REDE NEURONAL - CLASSIFICAÇÃO
# ============================================================

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

print("=" * 60)
print("4.3.1.b REDE NEURONAL - CLASSIFICAÇÃO")
print("=" * 60)

# ------------------------------------------------------------
# Variáveis independentes e alvo
# ------------------------------------------------------------

X = df_class.drop(
    columns=[
        "utilizRede",
        "Util_Decimal",
        "PFolga_PTD"
    ]
)

y = df_class["utilizRede"]

# Converter booleanos para 0/1
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

# Codificar classes: baixo, médio, alto -> números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes codificadas:")
for classe, codigo in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{classe} -> {codigo}")

# ------------------------------------------------------------
# Configurações da rede
# ------------------------------------------------------------

configs_nn_class = [
    {
        "nome": "Config 1 - Rede simples",
        "hidden_layer_sizes": (32,),
        "alpha": 0.0001,
        "learning_rate_init": 0.001
    },
    {
        "nome": "Config 2 - Rede intermédia",
        "hidden_layer_sizes": (64, 32),
        "alpha": 0.001,
        "learning_rate_init": 0.001
    },
    {
        "nome": "Config 3 - Rede profunda",
        "hidden_layer_sizes": (128, 64, 32),
        "alpha": 0.01,
        "learning_rate_init": 0.0005
    }
]

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

resultados_nn_class = []
historicos_loss_class = {}

melhor_f1 = -1
melhor_config_nome = None
melhor_y_true = None
melhor_y_pred = None

# ------------------------------------------------------------
# Treino e validação cruzada
# ------------------------------------------------------------

for config in configs_nn_class:

    print("\n" + "=" * 60)
    print(config["nome"])
    print("=" * 60)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        # Standardização dentro do fold
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=config["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=config["alpha"],  # regularização L2
            learning_rate_init=config["learning_rate_init"],
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        accuracy_scores.append(acc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        print(
            f"Fold {fold} -> "
            f"Accuracy = {acc:.4f} | "
            f"Precision = {precision:.4f} | "
            f"Recall = {recall:.4f} | "
            f"F1-score = {f1:.4f}"
        )

        # Guardar curva de loss do primeiro fold de cada configuração
        if fold == 1:
            historicos_loss_class[config["nome"]] = model.loss_curve_

    acc_medio = np.mean(accuracy_scores)
    precision_media = np.mean(precision_scores)
    recall_medio = np.mean(recall_scores)
    f1_medio = np.mean(f1_scores)

    resultados_nn_class.append([
        config["nome"],
        config["hidden_layer_sizes"],
        config["alpha"],
        config["learning_rate_init"],
        acc_medio,
        precision_media,
        recall_medio,
        f1_medio
    ])

    print("-" * 60)
    print(f"Accuracy Média : {acc_medio:.4f}")
    print(f"Precision Média: {precision_media:.4f}")
    print(f"Recall Médio   : {recall_medio:.4f}")
    print(f"F1-score Médio : {f1_medio:.4f}")

    if f1_medio > melhor_f1:
        melhor_f1 = f1_medio
        melhor_config_nome = config["nome"]
        melhor_y_true = y_true_all.copy()
        melhor_y_pred = y_pred_all.copy()

# ------------------------------------------------------------
# Resumo das configurações
# ------------------------------------------------------------

resultados_nn_class = pd.DataFrame(
    resultados_nn_class,
    columns=[
        "Configuração",
        "Camadas",
        "Alpha L2",
        "Learning Rate",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]
)

print("\nResumo das Redes Neuronais - Classificação:")
print(resultados_nn_class)

print("\nMelhor configuração:")
print(f"{melhor_config_nome} | F1-score = {melhor_f1:.4f}")

# ============================================================
# CURVAS DE LOSS - REDE NEURONAL CLASSIFICAÇÃO
# ============================================================

for nome_config, loss_curve in historicos_loss_class.items():

    plt.figure(figsize=(8, 5))

    plt.plot(
        loss_curve,
        label="Loss Treino"
    )

    plt.title(f"Curva de Loss - {nome_config}")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# RELATÓRIO DA MELHOR REDE NEURONAL
# ============================================================

print("\nRelatório de Classificação - Melhor Rede Neuronal:")

print(
    classification_report(
        melhor_y_true,
        melhor_y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

cm = confusion_matrix(
    melhor_y_true,
    melhor_y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title(f"Matriz de Confusão - {melhor_config_nome}")
plt.xlabel("Classe prevista")
plt.ylabel("Classe real")
plt.tight_layout()
plt.show()
