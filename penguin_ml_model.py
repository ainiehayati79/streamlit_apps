import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

penguin_df = pd.read_csv("penguins.csv")
penguin_df.dropna(inplace=True)

output = penguin_df ['species']
features = penguin_df[['island','bill_length_mm','bill_depth_mm',
'flipper_length_mm','body_mass_g','sex']]

#print(output.tail())
#print(features.tail())
features = pd.get_dummies(features)
output, uniques = pd.factorize(output)

x_train, x_test, y_train, y_test = train_test_split(features, output, test_size =0.8)
random_state=123
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)
score = accuracy_score(y_pred, y_test)
print("Our accuracy score for this model is {}".format(score))
#print('succesfully! good job ainie')

#save the penquin RF RandomForestClassifier
rfc_pickle = open("random_forest_penguin.pickle", 'wb')
pickle.dump(rfc, rfc_pickle)
rfc_pickle.close()

output_pickle = open('outout_penguin.pickle','wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()
