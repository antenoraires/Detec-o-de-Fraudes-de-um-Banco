import pandas as pd 

def rename_dataframe(df):
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

    return  df.rename(columns= colunas)