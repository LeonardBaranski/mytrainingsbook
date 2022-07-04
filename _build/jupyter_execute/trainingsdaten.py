#!/usr/bin/env python
# coding: utf-8

# # Meine Trainingsdaten

# In dieser Datei, erstelle ich zwei Modelle. <br>
#  - Eines, welches anhand der angegebenen Geschwindigkeit und der Laufstrecke die Anstrengung während des Laufens vorhersagt, <br>
#  - und ein anderes, welche die drei Paramter `Strecke in km`, `m/s` und `Empfinden` annimmt, und somit die maximale durchschnittliche Herzfrequenz hervorsagt, <br>welche während diesem Lauf nicht überschritten werden sollte.

# ## Setup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers
import seaborn as sns

from tensorboard import notebook
import datetime

import sklearn.metrics
from sklearn.preprocessing import StandardScaler
import shap


tf.__version__


# In[2]:


# Laden der TensorBoard extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# ## Data

# Um das Modell zu trainieren werde ich meine Trainingsdaten aufteilen in die verschiedenen Trainingsbereiche, die Trainingsphasen, <br>
# die jeweiligen Wochenkilometer und die Geschwindigkeit spezifischer Ausdauerprogramme:
# 
# feature| Description| Feature Type
# ------------|--------------------|----------------------
# Strecke in km | Laufstrecke in KM | Numerical
# Empfinden | Die Empfundene Anstrengung während des Laufens auf einer Skala von 0 bis 10 | Numerical
# m/s | Die durchschnittliche Geschwindigkeit in m/s | Numerical
# Herzfrequenz | Die durchschnittliche Herzfrequenz | Numerical

# ### Data import

# - Hier Lade ich die Daten zunächst in einen Pandas Dataframe
# 
# - Welches Modell man jetzt Trainineren will, hängt davon ab, welchen Codeblock man laufen lässt.
# - Ich fokussiere mich in diesem Notenbook auf das Herzfrequenzmodell (Trainingsdaten_VS)

# In[3]:


#Einlesen der CSV
dfTR = pd.read_csv("Trainingsdaten_060522.csv", sep=",")


# In[4]:


dfTR = pd.read_csv("Trainingsdaten_VS.csv", sep=",")


# In[5]:


dfTR.info()


# ### Data Preparation

# - Zuerst wird die Spalte Woche aufgeteilt in `KW` und `Jahr`, damit man damit später auch rechnen kann

# In[6]:


#Woche in KW und Jahr unterteilen
dfWoche = dfTR["Woche"].str.split("/", n=2, expand=True)
dfTR["KW"] = dfWoche[0]
dfTR["Jahr"] = dfWoche[1]


# - Zur besseren Übersicht, werden vorerst unnötige Spalten aus dem Dataframe entfernt

# In[7]:


#Löschen der unnötigen Spalten
dfTR.drop(["Laktat", "Bemerkung", "Name"], axis=1, inplace=True)


# - Dann wird nach den Dauerläufen im Bereich `GA1` und `GA2/KA Schwelle` mit dem filter `flach (L)` sortiert, damit die Tempolaufprogramme aus dem Dataframe gefiltert werden
# - Diese Sortierung wird unserer finalen `df` variable zugewiesen

# In[8]:


#nur nach GA1 und GA2 schnelle Dauerläufe sortiert
df = dfTR[(dfTR["Bereich"] == "GA1") | (dfTR["Bereich"] == "GA2/KA Schwelle") & (dfTR["Kategorie"] == "flach (L)")]


# In[9]:


df.head()


# - Wie der überliegenden Tabelle zu entnehmen ist, ist der Dataframe noch nicht ganz lesbar für unser Modell
# - Die Spalte `Programm` wird also im folgenden aufgesplittet, und in mehrere Spalten unterteilt
# - Des weiteren wird auch die Spalte `HF` (Herzfrequenz) aufgeteilt

# In[10]:


# Programm in zwei gesplittet und Programm gelöscht
new = df["Programm"].str.split("|", n=1, expand=True)
df.drop(columns=["Programm"], inplace=True)

# Aus den ersten zwei teilen Strecke und Pace gemacht
df["Strecke in km"] = new[0]
df["Pace"] = new[1]

# Aus Pace nochmal zwei teile gemacht und Pace gelöscht
newkmh = df["Pace"].str.split("|", n=1, expand=True)
df.drop(columns=["Pace"], inplace=True)

# Meter pro sekunde und min/km als Spalte hinzugefügt
df["m/s"] = newkmh[0]
df["min/km"] = newkmh[1]

# HF wird in zwei gesplittet und die Spalte HF im Dataframe gelöscht
hf = df["HF"].str.split("/", n=1, expand=True)
df.drop("HF", axis=1, inplace=True)

# Die zwei Teile werden df hinzugefügt
df["AVG_HF"] = hf[0]
df["MAX_HF"] = hf[1]


# - In meinem Datenset sind bei einigen Feldern keine Herzfrequenz eingetragen
# - Diese müssen aussortiert werden

# In[11]:


df = df.drop(df[df["AVG_HF"] == "-"].index)


# In[12]:


df.info()


# In[13]:


df.tail()


# - Die Einheiten werden noch Entfernt
# - Und als nächstes werden die numerischen Felder in `float` und die textfelder in `str` umgewandelt

# In[14]:


#Entfernen der Einheiten und Umwandeln der Datenfelder in float
df["Strecke in km"] = df["Strecke in km"].str.replace(r"km", "").astype(float)
df["m/s"] = df["m/s"].str.replace(r"m/s", "").astype(float)
df["min/km"] = df["min/km"].str.replace(r"min/km", "").astype(str)


# In[15]:


df.head()


# In[16]:


df.info()


# ### Data format

# - Jetzt müssen wir unsere Daten noch ein weniger weiter formatieren

# - Die Spalte `AVG_HF` wird in int umgewandelt und `MAX_HF` wird entfernt

# In[17]:


df["AVG_HF"] = df["AVG_HF"].astype("int")
df.drop("MAX_HF", axis=1, inplace=True)


# - Aus Performancegründen wandeln wir folgende Werte um:
# 
#   - `int64` in `int32`
#   - `float64` in `float32` 

# In[18]:


# Erstellen eines dictionarys mit int64 Spalten als keys und np.int32 als werte
int_32 = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
# Alle spalten in dem dictionary verändern
df = df.astype(int_32)

# Erstellen eines dictionarys mit float64 Spalten als keys und np.float32 als werte
float_32 = dict.fromkeys(df.select_dtypes(np.float64).columns, np.float32)
df = df.astype(float_32)


# In[19]:


float_32


# In[20]:


df.info()


# - Als nächstes wandeln wir die kategorischen Daten in numerische um

# In[21]:


df["Empfinden"] = df["Empfinden"].replace(["0 - Ruhe", "1 - sehr leicht", "2 - leicht", "3 - moderat", "4 - etwas anstrengend", 
"5 - anstrengend", "6", "7 - sehr anstrengend", "8", "9 - extrem anstrengend", "10 - maximale ausbelastung"], ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]).astype(int)


# - Als nächstes erstelle ich eine weitere CSV, wo das datum noch enthalten ist, damit ich damit später in Streamlit arbeiten kann
# - Hier ist das Datum allerdings zunächst unnötig

# In[22]:


df.drop(["Bereich", "Kategorie", "Typ", "min/km", "Woche"], axis=1, inplace=True)
df.to_csv("cleaned_data.csv", index=False)


# - Hier entferne ich für eine bessere Übersicht, die temporär unwichtigen Spalten, damit sich für mein Modell nur meine vier wichtigen Variablen im df befinden

# In[23]:


df.drop(["KW", "Jahr", "Datum"], axis=1, inplace=True)


# - Die folgende csv ist für die Datenvisualisierung des Anstrengungsmodelles in Streamlit da, und ist somit für einen Durchlauf des Herzfrequenzmodelles irrelevant 

# In[24]:


df.to_csv("anstregung.csv", index=False)


# In[25]:


df.head()


# - Der fertig präparierte Dataframe hat jetzt 46 Zeilen und 4 Spalten
# - Aufgrund meiner Datenlage kann man das Modell nicht als Aussagekräftig bezeichnen, jedoch ist die Funktion genau die selbe

# In[26]:


df.shape


# In[27]:


df.tail()


# ### Data Visualization

# - Als nächstes können wir uns die Daten visualisieren lassen.

# - Als generelle Übersicht vor einer Visualisierung ist es gut, einen Blick auf die Statistische Auswertung der Daten zu werfen. 
# - Dies funktioniert sehr gut mit der pandas funktion `Dataframe.describe()`

# In[28]:


df.describe()


# #### Visuelle Darstellung mit Seaborn

# - Mit Pandas lässt sich der Dataframe allerdings nicht visuell darstellen lassen. Dafür benutze ich Seaborn
# - Seaborn ist eine Python Datenvisualisierungsbibliothek, auf der Basis von matplotlib
# - Sie ermöglicht mit einfachen Befehlen einen sehr guten Überblick über eventuelle Abhängigkeiten zu erhalten

# - Hier bekommt man schon einen guten Überblick darüber, welche Features in Abhängigkeit stehen

# In[29]:


# Auflisten der vier features in sns.pairplot
pp = sns.pairplot(df[["Strecke in km", "m/s", "Empfinden", "AVG_HF"]])
pp = pp.map_lower(sns.regplot)
pp = pp.map_upper(sns.kdeplot)


# ### Data Splitting

# - Als ersten Schritt wird die zu vorhersagende Variable in einer Kopie mit der Variable `X` aus dem Dataframe entfernt
# - Als zweites wird eine weitere Kopie des Dataframes mit der Variable `y` initiiert, welche nur die target variable enthält

# In[30]:


# Dieses Codeelement wird nur für das Anstrengungsmodell benutzt
#X = df.drop(["Empfinden"], axis=1)
#y = df["Empfinden"]


# In[31]:


X = df.drop(["AVG_HF"], axis=1)
y = df["AVG_HF"]


# - Mit hilfe der sklearn-Methode `train-test-split()` werden jetzt die zwei Dataframes in vier verschiedene Variablen unterteilt
# - Hierbei beträgt die split-size 20% der Daten

# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[33]:


print(
    "%d Datensätze werden für das Training benutzt, und %d für die validation"
    %(len(X_train), len(X_test))
)


# ## Model

# - Ich benutze bei meinem Modell ein Sequentielles Modell
# - Dieses Modell benutzt man dann, wenn es einen Input und einen Output des Modells gibt
# 

# - Zuerst wird hier das Sequentielle keras Modell instanziiert

# In[34]:


model = tf.keras.Sequential()


# - In meinem Fall benutze ich 9 Knoten im ersten Layer, mit `input_dim=3`, da ich in meinem Test dem Modell 3 Input Features übergebe
# - Mein Output Layer hat dann nur noch einen Output, da ich nur versuche vorherzusagen, welche durschnittliche Herzfrequenz man während dem Dauerlauf nicht überschreiten sollte.

# In[35]:


# Erster Layer
model.add(tf.keras.layers.Dense(9, input_dim=3, activation="relu"))
# Zweiter Layer
model.add(tf.keras.layers.Dense(6, activation="relu"))
# Dritter Layer
model.add(tf.keras.layers.Dense(3, activation="relu"))
# Output Layer
model.add(tf.keras.layers.Dense(1, activation="linear"))


# In[36]:


model.weights


# In[37]:


model.summary()


# - Mit Keras Model.compile() wird das Modell dann konfiguriert.

# In[38]:


model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae"])


# - Hier wird für das Tensorbard das logs verzeichnis erstellt

# In[39]:


# Create TensorBoard folders
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# ##### Callbacks
# - Für mein Modell lege ich außerdem Callbacks fest. 
# - Callbacks sind dafür da, bestimme Aktionen in verschiedensten Phasen des Trainings durchzuführen.
# - In meinem Fall führe ich ein Tensorboard aus, um die Ergebnisse später gut visualisieren zu können
# - Des weiteren implementiere ich einen EarlyStopping Callback, um Overfitting zu vermeiden. 
# - Nachdem sich der `val_loss` nach 15 Durchläufen nicht verbessert hat, wird das Training unterbrochen

# In[40]:


my_callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
]


# ## Training

# - Hier wird das Modell dann der Variable test_history zugeordnet, trainiert und getestet
# - Ich habe mich in meinem normalen durchlauf und dem Training für 2000 Epochen nach etwas probieren entschieden, weil ich so auf augenscheinlich gute ergebnisse gestoßen bin

# In[41]:


test_history = model.fit(X_train, y_train, epochs=2000, validation_split=0.2, verbose=1, callbacks=my_callbacks)


# - Ich lasse mein Modell hier ziemlich oft durchlaufen, und erziele dabei einen loss von 361
# - Um zu überprüfen, wie sich mein Modell verhält, und ob ich dies verwenden kann, werde ich beim Hyperparameter Tuning erfahren.

# ### Hyperparameter Tuning

# - Um mein Modell eventuell noch genauer zu machen, führe ich im Folgenden ein Hyperparameter Tuning durch
# - Mir geht es hierbei darum, herauszufinden, mit welchen Parametern ich mein Modell verbessern kann.
# 

# In[42]:


import keras_tuner as kt


# - Mithilfe der hp.Choice Funktion wird in dieser funktion beim Aufrufen jedes mal ein anderes Modell erzeugt.
# - Das Modell wird kompiliert und zurückgegeben

# In[43]:


def build_model(hp):
    model_hp = tf.keras.Sequential()
    model_hp.add(tf.keras.layers.Dense(
        hp.Choice('units', [8, 16, 32]),
        activation='relu'))
    model_hp.add(tf.keras.layers.Dense(1, activation='linear'))
    model_hp.compile(loss='mse', metrics=["mse", "mae"])
    return model_hp


# - Ich führe hier einen RandomSearch durch.
# - Dabei wird jede mögliche Kombination der verfügbaren Parameter durchprobiert.

# In[44]:


tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    project_name="hp_tuning"
    )


# - Mit meinen Callbacks ist diese Einstellung zu unruhig, und es kommt zu einem EarlyStopping
# - Der loss ist also höher, als bei meinem normal trainerten Modell.
# 
# - Dies suggeriert, dass mein normal trainertes Modell Overfittet ist. 

# In[45]:


my_callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
]


# In[46]:


tuner.search(X_train, y_train, epochs=2000, validation_data=(X_test, y_test), callbacks=my_callbacks)
best_model = tuner.get_best_models()[0]


# - Mit `Patience=15` warte ich hier 15 Steps. Wenn das Modell dann nicht besser wird, bricht das Training ab.
# - Overfitting wird bei meinem Hyperparameter Modell also vermieden.
# - Mit einem loss von knapp 915 ist dieses Modell alles andere als genau, jedoch ist dies aufgrund der Datenmenge und Qualität nicht verwunderlich

# ### Modellwahl

# - Ich entscheide mich für mein finales Modell also für das Beste aus meiner RandomSearch. 
# - Dies liegt an erster Stelle der get_best_models funktion

# In[47]:


# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
# Get best model
best_model = models[0]


# ## Evaluierung 

# - Im Folgenden werde ich das Modell mit einem Tensorboard, Matplotlib und Shap etwas genauer unter die Lupe nehmen.

# - Laden einer Tensorboard instanz zur Visualisierung des Modells

# In[48]:


get_ipython().run_line_magic('tensorboard', '--logdir . --port=6007')


# In[49]:


notebook.list()


# - Mit matplotlib.pyplot kann man sich den Fortschritt und die Entwicklung über das Training des Modells genau anschauen, ohne dabei eine tensorboard instanz zu starten:
# - Den selben Überblick bekommt man jedoch natürlich nicht 
# - Hier bilde ich nur mein normal trainiertes modell ab, um dies mit dem tensorboard modell vergleichen zu können

# In[50]:


history_df = pd.DataFrame(test_history.history)
plt.plot(history_df["loss"], label="loss")
plt.plot(history_df["val_loss"], label="val_loss")

plt.legend()


# ### SHAP

# - Mit SHAP lässt sich ein Modell sehr gut auswerten. 
# - Man kann hier visualisieren, welche Werte sich wie stark auf das Modell auswirken.
# - So fällt es einem leichter, die Funktionsweise des Modells und die Hintergründe verschiedenster Predictions leichter zu erkennen

# In[61]:


shap.initjs()


# - Hier ordne ich dem Shap Explainer zuerst eine Variable zu.
# - Der Shap Explainer ist das primäre Explainer Interface von Shap. 
# - Diesem übergeben wir unser Modell und unsere Daten

# In[62]:


explainer = shap.Explainer(best_model.predict, X)


# - Der `shap_values` variable übergeben wir jetzt das explainer Objekt mit unseren Daten

# In[63]:


shap_values = explainer(X)


# - Als erstes lasse ich hier einen Bar Plot mit dem Mittelwert des sogenannten `SHAP values` ausgeben
# - Dieser Wert zeigt die Auswirkung des Wertes auf das Modell an.
# - Je höher der Wert, desto größer der Einfluss dieses Features auf die Prediction des Modells

# In[66]:


shap.plots.bar(shap_values)


# - Was bei meinem Modell heraussticht, ist vor allem erst die `Strecke in km`
# - Dies ist auch das erste Learning bei meiner Modellanalyse.
# - Je länger die Strecke, desto höher die durchschnittliche Herzfrequenz. 
# - Mathematisch macht dies auch Sinn, da bei einer längeren Belastung der Schnitt im allgemeinen höher ist. 
# - Würde man bei einer 30 min Belastung auf einen Puls von 180 kommen und diesen halten, hat man natürlich dann bei einer 60 minuten Belastung einen höheren Gesamtschnitt.
# - Der Wert dieser Variable ist also sehr wichtig für die Prediction, wie man auch im folgenden Summary Plot erkennen kann

# In[67]:


shap.summary_plot(shap_values)


# - Der Violin Plot ist eigentlich der selbe, wie deer Summary Plot, jedoch finde ich die Darstellung noch etwas anschaulicher, da man einen besseren Überblick bekommt, wo sich die meisten Werte ansiedeln.
# - Auch das `Empfinden` hat einen ziemlich hohen SHAP value.
# - Dies lässt mich darauf schließen, dass die Herzfrequenz weniger von der Geschwindigkeit, sondern eher von der Tagesform und dem Gefühl beim Laufen abhängig ist.
# - Dieser Fakt macht es natürlich deutlich schwieriger Leistungen vorherzusagen, da man diese nicht an der Geschwindigkeit messen kann, sondern an ganz vielen anderen Faktoren

# In[68]:


shap.summary_plot(shap_values, plot_type="violin")


# ## Testing 

# - Als letztes gilt es das Modell zu testen.
# - Mit Shap und dem vorigen Kapitel haben wir dies ja auch schon getan.
# - In diesem Kapitel werden noch ein paar Zahlen gezeigt, das Modell ausgetestet und außerdem abgespeichert, damit ich in der app.py darauf zugreifen kann
# 
# - Mit der .evaluate funktion kann man das Modell gut auswerten

# In[71]:


best_model.evaluate(X_test, y_test, batch_size=16)


# - Auch ein Graph kann helfen, die Genauigkeit eines Modells abzubilden, indem man die echten Werte mit den vorhergesagten Vergleicht:

# In[72]:


y_pred = model.predict(X_test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('A plot that shows the true and predicted values')
plt.xlim([0, 200])
plt.ylim([0, 200])
plt.plot([0, 200], [0, 200])


# - Hier lasse ich mir kurz einen Auszug aus dem Dataframe geben, damit ich die Reihenfolge der zu übergebenden Variablen im testing sehen kann

# In[73]:


df.head()


# - Als nächtes generiere ich einen test input mit erfundenen Variablen für die Werte [`Empfinden`, `Strecke in km`, `m/s`]

# In[74]:


new_input = [[3, 8, 3.7]]


# - Mit diesem test input kann ich jetzt das Modell testen, um zu sehen, welche `AVG_HF` vorhergesagt wird
# - Dies hilft mir dabei zu visualisieren, wie der Benutzer später in Streamlit die Daten übergeben sollte

# In[75]:


new_output = best_model.predict(new_input)
print(new_output)


# - Das Modell wird abgespeichert, damit es in der app.py direkt aufgerufen werden kann, ohne weitere Berechnungen durchführen zu müssen
# - Ich speichere hier außerdem auch mein altes Modell vor dem Hyperparameter tuning, um für mich Vergleichswerte zu haben

# In[76]:


best_model.save("best_model")


# In[77]:


model.save("hf_model")


# - Dasselbe gilt für das Anstrengungsmodell, falls man dieses bearbeiten möchte.

# In[44]:


model.save("feeling_model")


# In[45]:


model = tf.keras.models.load_model("feeling_model")

