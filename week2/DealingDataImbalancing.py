      # remove bad test cases from test dataset
       Test_size = 0.20
        y_series = pd.Series(y)
        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]
        y_bad = y[y_series.isin(good_y_value) == False]
        X_bad = X[y_series.isin(good_y_value) == False]
       test_size = X.shape[0] * 0.2 / X_good.shape[0]
        print(f"new_test_size: {new_test_size}")
        X_train, X_test, y_train, y_test = train_test_split(X_good, y_good,     test_size=test_size, random_state=0)
        X_train = np.concatenate((X_train, X_bad), axis=0)
        y_train = np.concatenate((y_train, y_bad), axis=0)
