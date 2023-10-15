from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# set data
iris = datasets.load_iris()
features = pd.DataFrame(iris["data"], columns=iris["feature_names"])
target = iris["target"]

# model
x_train, x_test, t_train, t_test = train_test_split(features, target, test_size=0.3, random_state=0)
model = RandomForestClassifier()
model.fit(x_train, t_train)
print(f"訓練データのaccuracy : {model.score(x_train, t_train)}")
print(f"検証データのaccuracy : {model.score(x_test, t_test):.2f}")

# keep
import pickle
pickle.dump(model, open("model_iris", "wb"))