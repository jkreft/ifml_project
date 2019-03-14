from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

class GBM(MultiOutputRegressor):
    def __init__(self, action_space, state_space):
        super().__init__(LGBMRegressor(n_estimators=100, n_jobs=-1))
        
        #self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))
        self.isFit = False
        
        self.action_space = action_space
        self.state_space = state_space 
        
        
class RegressionForest(RandomForestRegressor):
    def __init__(self):
        super().__init__(n_estimators=100)
        
        
        
class DecisionTree(DTC):
    def __init__(self):
        super()
        
    
class QTable():
    def __init__(self):
        passs