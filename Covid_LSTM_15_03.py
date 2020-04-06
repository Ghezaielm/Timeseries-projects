#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot  as plt 
import pandas as pd 
import os 
import urllib.request
from pytrends.request import TrendReq
from datetime import date, time, datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[5]:


class covid_analysis(): 
    
    def __init__(self): 
        self.path = '/home/jaz/Documents'
        self.keywords = ['Coronavirus','Covid','Covid-19','Horaire','Train', 
                        'pharmacie','laposte','bus','train','tram','metro', 
                        'cinema','velib','restaurant']
        
        self.alimentaire = ['Coronavirus','Restaurant','Pizzeria','Tacos','Carrefour','Auchan','Lidl','Leaderprice', 
                           'monoprix','monop']
        self.logistique = ['train','bus','tram','metro','velib']
        self.administratif = ['mairie','pôle emploi','laposte']
        self.education = ['assistante',"Garde enfant","maternelle",'Ecole',"collège","Lycée",'université','fac']
        self.sport = ['Piscine','Stade']
        self.culturel = ['cinema','café','bar','tabac','bibliothèque','mediathèque','exposition']
        self.religion = ['Eglise','Temple','Synagogue','Mosquée','Masjid']
        self.sanitaire = ['Pharmacie','Hopitaux','Ehpad']
        self.risk = ['Masque','Gel','Malade','Rhume','Diarhée','Toux','Mouchoirs','Fievre']
        self.immobilier = ['Appartement','Studio','Location','immobilier','louer','airbnb','maison']
        self.politique = ['Macron','Election','Gouvernement','Annonce']
        self.travail = ['Arret travail','Chomage','Chomage partiel','Chômage']
        self.others = ["Transmission","Mains",'Bises','Virus','Bactéries']
        self.key_words = [self.alimentaire,self.logistique,self.administratif,self.education,
                         self.sport,self.culturel,self.religion,self.sanitaire,self.risk,self.immobilier, 
                         self.politique,self.travail,self.others]
        
        self.key_words = [i[j] for i in self.key_words for j in range(len(i))]
        
        print(self.key_words)
    def get_timeframe(self):
        self.delay = 10
        date = datetime.now()
        start_behav = date - timedelta(days=self.delay)
        return "2020-01-22 2020-{}-{}".format( date.month,date.day)
    
    
    def get_data(self): 
        
        # Get the case file 
        urllib.request.urlretrieve('https://www.data.gouv.fr/fr/datasets/r/f4935ed4-7a88-44e4-8f8a-33910a151d42',
                                   os.path.join(self.path,"world_cases.csv"))
        # Get the by region case in france area 
        urllib.request.urlretrieve('https://www.data.gouv.fr/fr/datasets/r/fa9b8fc8-35d5-4e24-90eb-9abe586b0fa5',
                                   os.path.join(self.path,"by_region_cases.csv"))
        
        self.metadata = pd.DataFrame()
        # Get data from google search volumes : 
        if os.path.exists(os.path.join(self.path,"metadata.csv"))==False:
            for i in self.key_words:
                print(i)
                # Login to Google. Only need to run this once, the rest of requests will use the same session.
                pytrend = TrendReq()

                # Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()

                tf = covid_analysis.get_timeframe(self)
                print(self.key_words.index(i),"/",len(self.key_words))
                pytrend.build_payload(kw_list=[i],timeframe=tf, geo='FR')
                df = pytrend.interest_over_time()
                
                self.metadata[i]=df[i]
            self.metadata.to_csv(os.path.join(self.path,"metadata.csv"))
        else: 
            self.metadata = pd.read_csv(os.path.join(self.path,"metadata.csv"))
        
        self.metadata = pd.DataFrame(self.metadata)
        self.world_cases_path = os.path.join(self.path,'world_cases.csv')
        self.by_region_cases_path = os.path.join(self.path,"by_region_cases.csv")
        

    
    def load_data(self):
        # Load the cases dataset
        self.data = pd.read_csv(self.world_cases_path,skiprows=3,sep=";")
        print(self.data)
        
    def analyze_data(self): 
        # Keep the most corrlated features 
        import seaborn as sns
        corr = self.metadata.corr()
        corr = corr[abs(corr['Coronavirus'])>0.2]
        corr = corr.T
        corr = corr[abs(corr['Coronavirus'])>0.2]
        corr = corr.sort_values(['Coronavirus'], ascending=[1],axis=0) 
        ax = sns.heatmap(corr)
        plt.show()
        self.feature_to_track = [i for i in corr.index.values]+[i for i in corr.columns.values] 
        self.feature_to_track = np.unique(self.feature_to_track)
        print("Feature to track: ",self.feature_to_track)
        
        # Filter the dataset: 
        self.metadata = self.metadata.filter(self.feature_to_track,axis =1)
        
    def show_data(self,country):
        # Correlation of features with the coronavirus keyword
        self.curr = self.data[self.data["Pays"]==country].iloc[::-1]
        f = plt.figure(figsize = (15,10))
        f.suptitle(country,fontsize =42)
        self.cols = [i for i in self.curr.columns.values[2:]]
        plt.style.use('dark_background')
        print(self.data['Date'])
        def gamma(a,b): 
            return ((a+2)/a)*b
        for idx,i in enumerate(self.cols):
            ax = f.add_subplot(231+idx)
            ax.set_title(i,fontsize=15)
            ax.plot(self.curr[i].values,c='red')
        plt.show()
        
    def process_data(self):
        
        # Define the data
        self.cases = self.data[self.data["Pays"]=='France'].iloc[::-1]
        self.metadata = self.metadata.iloc[:-1,:]
        
        # Define columns
        cases_names = {idx:i for idx,i in enumerate(self.cases.columns.tolist()[2:])}
        metadata_names = {idx:i for idx,i in enumerate(self.metadata.columns.tolist())}
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        
        # Store the data in a dataframe
        self.metadata = pd.DataFrame(scaler.fit_transform(self.metadata))
        self.cases = pd.DataFrame(scaler.fit_transform(self.cases.iloc[:,2:]))
        self.metadata = self.metadata.rename(columns=metadata_names)
        self.cases = self.cases.rename(columns=cases_names)
        # We aim to predict the effect of social behaviour on the infection
        # As the infection is fast, we will use a windows of 3 days
        X = []
        y = []
        ws = 10
        n_steps_out = 10
        for idx in range(ws,self.cases.shape[0]-n_steps_out):
            X.append(self.metadata.iloc[idx-ws:idx,:].values)
            y.append(self.cases.iloc[idx:idx+n_steps_out,:].values)
        print("X",np.array(X).shape)
        print("Y",np.array(y).shape)
        self.X = np.array(X)
        self.y = np.array(y)
        self.ws = ws
        self.n_out = n_steps_out
        
    def model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1,
                                                            random_state=42)
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM, TimeDistributed, RepeatVector
        from keras.layers.embeddings import Embedding
        
        # And we create the model
        self.model = Sequential()
        self.model.add(LSTM(200, activation='relu', input_shape=(self.ws, self.X.shape[2])))
        self.model.add(RepeatVector(self.ws))
        self.model.add(LSTM(200, activation='relu', return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.y.shape[2])))
        self.model.compile(optimizer='adam', loss='mse',metrics=["mae"])
        
        # Store the model as an attribute of the experiment object
        self.history= self.model.fit(X_train, y_train, epochs=200,
                                 batch_size=10,verbose=1)
        yhat = pd.DataFrame(self.model.predict(X_test, verbose=1)[0])
        f = plt.figure(figsize=(15,10))
        
        f.suptitle("Covid-19 metrics in France",fontsize=20)
        print(self.cases.shape)
        for idx,i in enumerate(self.cols):
            ax = f.add_subplot(231+idx)
            ax.set_title(i,fontsize=15)
            ax.plot(self.cases[i].values,c='blue')
            new = [j for j in self.cases[i]][:-self.ws]
            test = len(new)
            new = new + [i for i in yhat[idx]]
            ax.plot(new,c='red')
            ax.vlines(x=test,ymin=0,ymax = max(self.cases[i]),color='green',ls="dashed")

o = covid_analysis()
o.get_data()
o.load_data()
#o.process_data2()
o.analyze_data()
o.show_data('France')
o.show_data("Italie")
o.show_data("Allemagne")
o.show_data("Suisse")
o.show_data("Belgique")
o.show_data("Espagne")
o.show_data("Italie")
o.show_data("Andorre")
o.process_data()
o.model()


# In[111]:




