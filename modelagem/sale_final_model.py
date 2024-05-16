"""
Descrição do Script
Este script é responsável pelo processo de treinamento de um modelo de regressão para prever os preços de venda de imóveis.
Utiliza o algoritmo XGBoost, que é ajustado com base nos melhores hiperparâmetros encontrados através de uma busca em grade.
Os dados são carregados, processados para converter categorias em variáveis dummy, e em seguida, o modelo é treinado.
O modelo treinado é salvo para uso futuro em avaliações ou produção.
"""

# Importando Bibliotecas
import joblib  
from xgboost import XGBRegressor  
import pandas as pd  
import ast  


# Carrega os dados de venda dos imóveis de um arquivo CSV.
df_sale = pd.read_csv("data/df_sale_final.csv", index_col=0)

# Carrega os resultados da busca em grade dos hiperparâmetros do modelo XGBoost.
xgb_grid_results = pd.read_csv("modelagem/grid_boosting_results_sale.csv", index_col=0)

# Extrai o melhor conjunto de hiperparâmetros para o modelo XGBoost do resultado da busca em grade.
best_xgb_result = ast.literal_eval(xgb_grid_results.sort_values("rank_test_score")["params"].reset_index(drop=True)[0])

# Transforma as colunas categóricas 'district_class' em colunas dummy, facilitando seu uso em modelos de aprendizado de máquina.
df_sale = pd.get_dummies(df_sale, columns=['district_class'], drop_first=True)

# Separa as variáveis independentes e dependentes para o treinamento do modelo.
X = df_sale.drop('Price', axis=1)  # Variáveis independentes.
y = df_sale['Price']  # Variável dependente, que é o preço do imóvel.

# Instancia o modelo XGBoost com os melhores hiperparâmetros encontrados.
xgb_final = XGBRegressor(random_state=42,
                         colsample_bytree=best_xgb_result["colsample_bytree"],
                         learning_rate=best_xgb_result["learning_rate"],
                         max_depth=best_xgb_result["max_depth"],
                         n_estimators=best_xgb_result["n_estimators"],
                         subsample=best_xgb_result["subsample"])

# Treina o modelo usando os dados de venda de imóveis.
xgb_final.fit(X, y)

# Salva o modelo treinado no disco para uso posterior.
joblib.dump(xgb_final, "modelagem/xgb_sale_final.pkl")

