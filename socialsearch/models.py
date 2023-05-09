from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from processing import CaptionProcessingTransformer


class ModelPipeline:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model
        return self
    
    def get_model(self):
        return self.model

    def dump_model(self, path):
        # export model
        raise NotImplementedError()
    
class SponsorClfXGBPipeline(ModelPipeline):
    def __init__(self, folds=5, random_state=None):
        self.folds = folds + 1 # k-fold cv + test set
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = (None,)*4

    def __ttsplit(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=(1/self.folds), 
            random_state=self.random_state, 
            stratify=y
        )

    def __pipeline(self):
        clf = XGBClassifier(n_estimators=20, max_depth=2, learning_rate=0.1, objective='binary:logistic')
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", MinMaxScaler()),
                ("clf",clf),
            ]
        )
        return pipe        

    def __full_pipeline(self, pipe):
        # add processing and gridsearchcv to model pipeline
        params = {
            "clf__n_estimators":[2, 5, 10 ,20, 40, 80],
            "clf__max_depth": [2,3,4],
            "clf__learning_rate":[0.1, 0.5, 1],
        }
        grid_pipe = Pipeline(
            [
                ("processing", CaptionProcessingTransformer()),
                ('grid_pipeline', GridSearchCV(pipe, param_grid=params, cv=self.folds-1, refit=True)),
            ]
        )        
        return grid_pipe

    def fit(self, X, y):
        self.__ttsplit(X, y)
        pipe = self.__pipeline() # pipeline
        full_pipe = self.__full_pipeline(pipe) #CV
        full_pipe.fit(self.X_train, self.y_train)

        gscv = full_pipe[1]
        print("Best Params: ", gscv.best_params_)
        print("Best Accuracy: ", gscv.best_score_)
        self.set_model(full_pipe) 
        
        return self

    def predict(self, X_new):
        if self.model == None:
            raise Exception("Fit not called, no model trained")
        
        return self.model.predict(X_new)       

    def evaluate(self):
        if self.model == None:
            raise Exception("Fit not called, no model trained")

        yhat = self.model.predict(self.X_test) 

        print(confusion_matrix(self.y_test, yhat))
        print(accuracy_score(self.y_test, yhat))
        print(f1_score(self.y_test, yhat))
        print(recall_score(self.y_test, yhat))