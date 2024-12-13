from detecao_fraude import Pipeline
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn import  metrics

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


fraude = Pipeline(x = x , y = y)

regression =  fraude.print()