from lofo.flofo_importance import FLOFOImportance
from data.test_data import generate_test_data
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


def test_flofo_importance():
    df = generate_test_data(1000)

    features = ["A", "B", "C", "D"]
    target = 'binary_target'
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)

    lgbm = LGBMRegressor(random_state=0, n_jobs=4)
    lgbm.fit(X_train, X_test)

    lofo = FLOFOImportance(lgbm, X_test, y_test)

    importance_df = lofo.get_importance()

    assert len(features) == importance_df.shape[0], "Missing importance value for some features!"
    assert importance_df["feature"].values[0] == "B", "Most important feature is different than B!"
