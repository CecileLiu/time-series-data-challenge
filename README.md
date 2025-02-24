## Time Series Data Analysis  
  
### Deal With Missing Value  
1. Numerical value imputation: `skmice_linear` accept `scikit-learn` models which are  `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `BayesianRidge`, `SGDRegressor`, `SVR`, `KNeighborsRegressor`; `skmice_tree` accept `scikit-learn` tree-based models.   

    ```
    # # skmice_linear usage example 

    import pandas as pd
    from imputation.skmice_linear import SKMiceImputer

    # SKMiceImputer + LinearRegression
    from sklearn.linear_model import LinearRegression
    # df = ... # load value from .csv or .xlsx
    X = df.values
    lr = LinearRegression()
    imputer1 = SKMiceImputer()
    seeded_X, specs = imputer1.transform(X, model_class=lr) # seeded_X is the imputed value

    # SKMiceImputer + KNeighborsRegressor
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor()
    imputer2 = SKMiceImputer()
    seeded_X, specs = imputer2.transform(X, model_class=knn)


    # # skmice_tree usage example
    from imputation.skmice_tree import SKTreeMiceImputer

    # SKTreeMiceImputer + RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()
    imputer3 = SKTreeMiceImputer()
    seeded_X, specs = imputer3.transform(X, model_class=rf)

    ```  
2. Categorical value imputation: set `strategy` to `most_frequent` in `SKMiceImputer` of `skmice_linear`  
    ```
    import pandas as pd
    from imputation.skmice_linear import SKMiceImputer
    from sklearn.preprocessing import OrdinalEncoder

    # .... # encode the categorical value to float or integer by OrdinalEncoder
    imputer4 = SKMiceImputer(strategy="most_frequent")
    X = df2[col_need].values
    seeded_X, specs = imputer4.transform(X, model_class=lr)
    # inverse transform
    inversed_imputed_value = enc.inverse_transform(seeded_X[:,-1].reshape(-1,1))

    ```