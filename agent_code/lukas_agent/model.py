from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

class GBM(MultiOutputRegressor):
    def __init__(self, args):
        self.name = "GB"
        super().__init__(LGBMRegressor(**args))
        #self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))
        
class RegressionForest(RandomForestRegressor):
    def __init__(self):
        self.name = "RF"
        super().__init__(n_estimators=100)
        
        
        
class DecisionTree(DTC):
    def __init__(self):
        self.name = "DT"
        super()
        
    
class QTable():
    def __init__(self):
        passs