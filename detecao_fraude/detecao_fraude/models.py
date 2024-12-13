from sklearn.linear_model import LogisticRegression
from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_test(x,y):
    SEED = 42
    x_train ,x_test,y_train ,y_test = train_test_split(x,y,
                                                    test_size= 0.25 , random_state= SEED)
    return x_train ,x_test,y_train ,y_test

def regressao(x_train , y_train, x_test):
    lr = LogisticRegression(max_iter= 1000 , random_state= 42)
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)
    return y_pred

def decision_tree(x_train , y_train, x_test):
    dt = DecisionTreeClassifier(max_depth= 5,  random_state= 42)
    model = dt.fit(x_train, y_train)
    y_pred =dt.predict(x_test)
    return y_pred

def random_forest(x_train , y_train, x_test):
    rf = RandomForestClassifier(max_depth= 5,  random_state= 42)
    model = rf.fit(x_train, y_train)
    y_pred =rf.predict(x_test)
    return y_pred