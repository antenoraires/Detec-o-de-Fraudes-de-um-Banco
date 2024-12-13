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

regression_pred =  fraude.regression()

# Registrando métricas no MLflow
mlflow.log_metric("final_accuracy", history.history['accuracy'][-1])
mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])


print("Acurácia:",metrics.accuracy_score(y_test, regression_pred))
print("Precisão:",metrics.precision_score(y_test, regression_pred))
print("Recall:",metrics.recall_score(y_test, regression_pred)) 
print("F1:",metrics.f1_score(y_test, regression_pred))

# Salvando o modelo no MLflow
mlflow.tensorflow.log_model(model, "modelo_moda")

# Encerrando a execução do MLflow
mlflow.end_run()

# Opcional: Salvar o modelo localmente também
model.save('model/modelo_moda.h5')