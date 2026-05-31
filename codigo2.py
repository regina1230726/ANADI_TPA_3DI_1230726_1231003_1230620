import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    "Nº omissos"   : df.isnull().sum(),
    "% omissos"    : (df.isnull().sum() / len(df) * 100).round(2)
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
    plt.figure(figsize=(12, 5))
    order = df['Tipo Construtivo'].value_counts().index
    
    # Gráfico de barras horizontal - corrigido sem o 'legend=False'
    sns.countplot(data=df, y='Tipo Construtivo', order=order, hue='Tipo Construtivo', palette='viridis')
    plt.title('Frequência dos PTDs por Tipo Construtivo')
    plt.xlabel('Contagem de Registos')
    plt.ylabel('Tipo Construtivo')
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


# Secção 4.1.3 - Pré-processamento dos Dados

    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder

    # Criar cópia do dataset original
    df_processed = df.copy()

# 1. Tratamento de Valores Omissos

    print("=" * 60)
    print("TRATAMENTO DE VALORES OMISSOS")
    print("=" * 60)

    # Verificar valores omissos
    missing_values = df_processed.isnull().sum()

    # Mostrar apenas colunas com omissos
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("Não existem valores omissos no dataset.")
    else:
        print(missing_values)

    # Variáveis numéricas -> substituir pela mediana
    num_cols = df_processed.select_dtypes(include=np.number).columns

    for col in num_cols:
        if df_processed[col].isnull().sum() > 0:
            mediana = df_processed[col].median()
            df_processed[col].fillna(mediana, inplace=True)

    # Variáveis categóricas -> substituir pela moda
    cat_cols = df_processed.select_dtypes(include='object').columns

    for col in cat_cols:
        if df_processed[col].isnull().sum() > 0:
            moda = df_processed[col].mode()[0]
            df_processed[col].fillna(moda, inplace=True)

    print("\nValores omissos após tratamento:")
    print(df_processed.isnull().sum().sum())

# 2. Transformação de Variáveis Categóricas

    print("\n" + "=" * 60)
    print("TRANSFORMAÇÃO DE VARIÁVEIS CATEGÓRICAS")
    print("=" * 60)

    # Verificar variáveis categóricas
    print("Variáveis categóricas encontradas:")
    print(cat_cols)

    # Aplicar Label Encoding
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

    print("\nTransformação concluída.")

# 3. Seleção de Variáveis Relevantes

    print("\n" + "=" * 60)
    print("SELEÇÃO DE VARIÁVEIS")
    print("=" * 60)

    # Remover variáveis pouco relevantes ou redundantes
    # (ajustar conforme análise do grupo)

    colunas_remover = [
        # exemplo:
        # 'CodDistritoConcelho'
    ]

    colunas_existentes = [c for c in colunas_remover if c in df_processed.columns]

    df_processed.drop(columns=colunas_existentes, inplace=True)

    print("Variáveis removidas:")
    print(colunas_existentes)

# 4. Normalização / Standardização

    print("\n" + "=" * 60)
    print("NORMALIZAÇÃO / STANDARDIZAÇÃO")
    print("=" * 60)

    # Standardização das variáveis numéricas
    scaler = StandardScaler()

    num_cols = df_processed.select_dtypes(include=np.number).columns

    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

    print("Standardização concluída.")

# 5. Dataset Final

    print("\n" + "=" * 60)
    print("DATASET FINAL PRONTO PARA MODELAGEM")
    print("=" * 60)

    print(f"Dimensão final: {df_processed.shape}")

    print("\nPrimeiras linhas do dataset processado:")
    print(df_processed.head())
    
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

# Preparação das matrizes (usando o df com dados limpos/imputados, mas SEM standardização global)
# Garantimos que os nulos já foram tratados conforme o teu código da secção 4.1.3
X_simple = df[[variavel_explicativa]].fillna(df[variavel_explicativa].median()).values
y_simple = df['PFolga_PTD'].fillna(df['PFolga_PTD'].median()).values

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
plt.scatter(X_scaled_vis[sample_indices], y_simple[sample_indices], alpha=0.5, color='teal', label='Dados Observados (Amostra)')

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