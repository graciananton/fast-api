from sklearn.ensemble import RandomForestRegressor
def create_forest(predictors, labels):
    forest_reg = create_random_forest_model(predictors,labels)
    return forest_reg

def create_random_forest_model(predictors,labels):
    forest_reg = RandomForestRegressor(
        bootstrap=True, 
        criterion= 'absolute_error', 
        max_depth=12, 
        max_features=3, 
        min_samples_split=6, 
        n_estimators=109
        )
    forest_reg.fit(predictors, labels)
    return forest_reg
