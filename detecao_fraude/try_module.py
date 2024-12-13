from detecao_fraude import Pipeline
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn import  metrics
import mlflow
import mlflow.tensorflow

from detecao_fraude.models import train_test

path = 'detecao_fraude/assets/data/fraud_dataset_example.csv'
df = pd.read_csv(path)

# renomeando df
df = df[['isFraud','isFlaggedFraud','step',
                            'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
                            'nameDest', 'oldbalanceDest', 'newbalanceDest', ]]

colunas = {
'isFraud': 'fraude',
'isFlaggedFraud':'super_fraude',
'step':'tempo',
'type':'tipo',
'amount':'valor',
'nameOrig':'cliente1',
'oldbalanceOrg':'saldo_inicial_c1',
'newbalanceOrig':'novo_saldo_c1',
'nameDest':'cliente2',
'oldbalanceDest':'saldo_inicial_c2',
'newbalanceDest':'novo_saldo_c2',
}

df = df.rename(columns= colunas)

# criando dummies
df = pd.get_dummies(data = df , columns= ["tipo"])
df = df.drop(['cliente1',"cliente2","super_fraude"], axis = 1)


y = df["fraude"] 
x = df.drop("fraude", axis = 1) 

mlflow.start_run()

x_train ,x_test,y_train ,y_test = train_test(x=x,y=y)
fraude = Pipeline(x = x , y = y)

# Modelo de regrssão 
print("Regressão Linear")
model_regre, regression_pred =  fraude.regression()

# Calcular o erro quadrático médio
mse = metrics.mean_squared_error(y_test, regression_pred)

# Registrar parâmetros e métricas no MLflow
mlflow.log_param("model_type", "Linear Regression")
mlflow.log_param("test_size", 0.2)
mlflow.log_metric("mse", mse)

# Registrar o modelo
mlflow.sklearn.log_model(model_regre, "linear_regression_model")

print("Métricas Regressão Linear")
print("Acurácia:",metrics.accuracy_score(y_test, regression_pred))
print("Precisão:",metrics.precision_score(y_test, regression_pred))
print("Recall:",metrics.recall_score(y_test, regression_pred)) 
print("F1:",metrics.f1_score(y_test, regression_pred))
print(".....................................................................................")

# modelo rando forest
print("Random Forest")
model_random, random_predit =  fraude.random()

mse_random = metrics.mean_squared_error(y_test, random_predit)
# Registrar parâmetros e métricas no MLflow
mlflow.log_metric("mse", mse_random)

# Registrar o modelo
mlflow.sklearn.log_model(model_random, "random_model")

print("Métricas Random Forest")
print("Acurácia:",metrics.accuracy_score(y_test, random_predit))
print("Precisão:",metrics.precision_score(y_test, random_predit))
print("Recall:",metrics.recall_score(y_test, random_predit))
print("F1:",metrics.f1_score(y_test, random_predit))
print(".....................................................................................")

print("Decision Tree")
model_tree, tree_predit =  fraude.decision()

tree_random = metrics.mean_squared_error(y_test, tree_predit)
# Registrar parâmetros e métricas no MLflow
mlflow.log_metric("mse", tree_random)

# Registrar o modelo
mlflow.sklearn.log_model(model_tree, "tree_model")

print("Métricas Decision Tree")
print("Acurácia:",metrics.accuracy_score(y_test, tree_predit))
print("Precisão:",metrics.precision_score(y_test, tree_predit))
print("Recall:",metrics.recall_score(y_test, tree_predit))
print("F1:",metrics.f1_score(y_test, tree_predit))


# Encerrando a execução do MLflow
mlflow.end_run()


