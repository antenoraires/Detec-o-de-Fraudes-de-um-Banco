from .models import regressao, decision_tree, random_forest, train_test # Certifique-se de importar random_forest
class Pipeline:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def print(self):
        return "print"

    def regression(self):
        """Executa a regressão e retorna os resultados."""
        self.x_train ,self.x_test, self.y_train ,self.y_test = train_test(self.x, self.y)
        model, result = regressao(self.x_train, self.y_train, self.x_test)
        return model , result

    def decision(self):
        """Executa a árvore de decisão e retorna os resultados."""
        self.x_train ,self.x_test,self.y_train ,self.y_test = train_test(self.x, self.y)
        model, result = decision_tree(self.x_train, self.y_train, self.x_test)
        return model, result

    def random(self):
        """Executa o random forest e retorna os resultados."""
        self.x_train ,self.x_test,self.y_train ,self.y_test = train_test(self.x, self.y)
        model ,result = random_forest(self.x_train, self.y_train, self.x_test)
        return model, result