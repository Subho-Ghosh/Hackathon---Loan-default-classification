# data manipulation
import pandas as pd
import numpy as np
import os,math,timeit,time
import matplotlib.pyplot as plt
import seaborn as sns

# data pre-post processing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold,RandomizedSearchCV
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel

# data modelling
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

# for automatic evoultion based feature selection
import feature_selection_ga as FSGA


class Process:
    '''Main class to do all data manipulations before and after modelling'''
    def __init__(self,filename,cols=[]):
        '''A generic function to load a given file, return a pandas dataframe'''  
        
        if len(cols)==0:
            # low_memory argument is used to limit the memory usage when trying to infer dtypes from a large dataset
            self.df = pd.read_csv(filename,low_memory=False) 
        else:            
            self.df = pd.read_csv(filename,low_memory=False,usecols = cols)
        
        # initialize few required variables
        self.cols_retained = self.df.columns
        self.predict_col = None
        self.target_col = None
        self.X = None
        self.y = None
        print("\n Initial File loaded with total {} columns and {} rows \n".format(self.df.shape[1],self.df.shape[0]))
    
    def setDF(self,indf):
        self.df=indf.copy()
        self.cols_retained = self.df.columns
    
    def getDF(self):
        return self.df.copy()
    
    def getShape(self):
        print("\n File has {} predictor columns and {} observation rows \n".format(len(self.cols_retained),self.df.shape[0]))
        
    def viewData(self,cols=[]):
        print("\n Data View \n")
        if len(cols)==0:
            print(self.df.head())
        else:
            print(self.df[cols].head())
    
    def viewInfo(self):
        print('\n Basic Metadata \n')
        print(self.df.info())  
    
    def viewMissingInfo(self):
        percent_missing = self.df.isnull().sum() * 100 / len(self.df)
        missing_value_df = pd.DataFrame({'column_name': self.df.columns,'percent_missing': percent_missing})
        missing_value_df.sort_values('percent_missing',ascending =False, inplace=True)   
        missing_value_df = missing_value_df[missing_value_df.percent_missing>0]
        
        print('\n Missing values information \n')
        if missing_value_df.shape[0]:
            print("\n There are {} cols with missing values(in descending order of nulls %age)\n".format(len(missing_value_df)))
            print(missing_value_df)
        else:
            print("\n no missing values found in any column \n")        
    
    def dropCols(self,cols=[]):
        self.cols_retained =[ c for c in self.cols_retained if c not in cols]        
    
    def getCols(self):
        return self.cols_retained
    
    def setCols(self,cols=[]):
        if len(cols)!=0:
            self.cols_retained = [c for c in self.cols_retained if c in cols]            
    
    def setTargetCol(self,target_col): # Its a file with only target defined
        self.target_col = target_col
        self.cols_retained = [c for c in self.cols_retained if c!=self.target_col]
    
    def setPredictCol(self,predict_col): # its the test file with no target columns
        self.predict_col = predict_col
        self.cols_retained = [c for c in self.cols_retained if c!=self.predict_col]
    
    def addTgtCol(self,best_model,model_cols):        
        self.df[self.target_col]=best_model.predict(self.df[model_cols])
        
    def addTgtColNN(self,best_model,model_cols):        
        self.df[self.target_col]=best_model.predict(self.df[model_cols])
        self.df[self.target_col] = (self.df[self.target_col]>0.5).astype(int)
        
    
    def cleanse_dedup(self,keylist = []):
        ''' remove full row or key based duplicates from the given dataframe    '''
        if not keylist:
            self.df.drop_duplicates(keep='first',inplace = True) 
        else:
            self.df.drop_duplicates(keep='first',subset = keylist,inplace = True)
            # remove any rows with null values for a key in keylist
            for key in keylist:
                self.df = self.df[self.df[key].isna()==False]        
    
    def imputeNulls(self):
        ''' Find and impute null/missing values based on data type'''
    
        nullcols = self.df[self.cols_retained].columns[self.df[self.cols_retained].isna().any()].tolist()   
    
        for c in nullcols:
            if self.df[c].dtype=='object':
                self.df[c] = self.df[c].fillna('')
            else:
                self.df[c] = self.df[c].fillna(0)
    
        print('\n List of columns with missing values now :',self.df[self.cols_retained].columns[self.df[self.cols_retained].isna().any()].tolist()) 
    
    def encode_cols(self,enc='label'):
        '''Encode all categorical columns - label or onehot'''
        #prepare list of all categorical variables in the dataset
        catlist = self.df[self.cols_retained].select_dtypes(include='object').columns.tolist()        
        
        if enc!='label':
            encoded=pd.get_dummies(self.df[catlist])
            # Drop columns as they are now encoded
            self.dropCols(catlist)
            self.df.drop(catlist,axis=1,inplace=True)
            #   Join the encoded df
            for c in encoded.columns.to_list():
                self.cols_retained.append(c)
            self.df = self.df.join(encoded)
        else:
            label_encoder = preprocessing.LabelEncoder() 
            self.df[catlist] = self.df[catlist].apply(label_encoder.fit_transform)
      
    def scale_cols(self,scaler='MinMax'):
        '''Scale continuous variables in data - MinMax or Standard''' 
        
        if scaler =='MinMax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
    
        #prepare list of all continuos variables in the dataset
        contvars = self.df[self.cols_retained].select_dtypes(exclude='object').columns.tolist()         
    
        self.df[contvars] = scaler.fit_transform(self.df[contvars])    
    
    def class_dist(self):
        '''Produce a graph of target class distribution for analysis '''
    
        df1 = pd.crosstab(index = self.df[self.target_col], columns = "count")
        df1['percent'] = df1/df1.sum() * 100
    
        print('\n Check target class distribution \n')
        print(df1)
    
        # graph of class distribution of the target variable
        df1.plot(kind='barh')
        plt.show()  
    
    def plot_corr(self):
        # Basic correlogram
        corr=self.df[self.cols_retained + [self.target_col]].corr()
        fig, ax = plt.subplots(figsize=(25,33))
        sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values,annot=True,ax=ax)
        plt.show()
    
    def plot_feature_imp(self):
        '''
        Plot a feautre importance graph for analysis
        '''
    
        self.clf = ExtraTreesClassifier(random_state=42,class_weight='balanced') # for repeatability of results
  
        self.X1 = self.df[self.cols_retained]
        self.y1 = self.df[self.target_col]
        self.clf.fit(self.X1,self.y1)    
    
        pd.Series(self.clf.feature_importances_, index=self.df[self.cols_retained].columns).plot(kind='bar',
                                                                                                 color='steelblue',
                                                                                                     figsize=(12, 18))
        plt.show()  
    
    def reduce_dims(self):        
        model = ExtraTreesClassifier(random_state=42,class_weight='balanced') # for repeatability of results
        # send only 5k rows for quick response
        fsga = FSGA.FeatureSelectionGA(model,self.df.loc[:8000,self.cols_retained].values
                                       ,self.df.loc[:8000,self.target_col])
        pop = fsga.generate(100)
        new_retained = []
        prev_cols = len(self.cols_retained)
        
        for ind,val in enumerate(pop):
            if val:
                new_retained.append(self.cols_retained[ind])
        
        self.cols_retained = new_retained[:]
        print("\n Columns retained after Genetic Selection Algorithm are: \n",self.cols_retained)
        print("\n {} columns retained out of {} \n".format(len(self.cols_retained),prev_cols))    
    
    def create_Xy(self,imbalance = 'N'):
        '''
        Separate the predictors from the target
        '''
        # assign variables and target data
        self.y = self.df[self.target_col]
        self.X = self.df[self.cols_retained]
    
        if imbalance == 'Y':
            # use SMOTE-Synthetic Minority Over-sampling Technique to balance out the target classes            
            sm = SMOTE(random_state=42)
            self.X,self.y = sm.fit_resample(self.X, self.y)
        else:
            self.X = np.array(self.X)
    
        print('\n No: of predictor variables is: {} and no:of observations for training is: {} \n'.format(self.X.shape[1]
                                                                                                          ,self.X.shape[0]))
        return self.X,self.y,self.cols_retained
    
    def PrintCSV(self,outfile):
        self.df[[self.predict_col,self.target_col]].to_csv(outfile,index=False)
        print("\n Prediction file written to disk \n")