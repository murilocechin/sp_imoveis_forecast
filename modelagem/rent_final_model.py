"""
Scrip da modelagem final de aluguel.
"""
# Importando Bibliotecas
import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import ast

# Lendo base de dados
df_rent = pd.read_csv("data/df_rent_final.csv", index_col=0)

# Obtendo os hiperparâmetros otimizados
rf_grid_results = pd.read_csv("modelagem/grid_rf_results_rent.csv", index_col=0)

# Ajustando melhor hiperparametro para um dicionario
best_rf = ast.literal_eval(rf_grid_results.sort_values("rank_test_score")["params"].reset_index(drop=True)[0])

# Codificando coluna categorica com one-hot-encoding
df_rent = pd.get_dummies(df_rent, columns=['district_class'], drop_first=True)

# Ajustando 
X = df_rent.drop('Price', axis=1)
y = df_rent['Price']

# Treinando modelo final
rf_final = RandomForestRegressor(random_state=42,
                                 max_depth=best_rf["max_depth"],
                                  min_samples_leaf=best_rf["min_samples_leaf"], 
                                  min_samples_split=best_rf["min_samples_split"], 
                                  n_estimators=best_rf["n_estimators"]
                                  )

rf_final.fit(X,y)

# Salvando modelo treinado
joblib.dump(rf_final, "modelagem/rf_rent_final.pkl")





