import numpy as np
import pandas as pd
from keras.models import load_model

# these are dummy models
class MLModel():
    def __init__(self) -> None:
        pass
    
    def predict(self, X):
        pred = self.model.predict(X)
        return (pred > 0.5).astype(int)
    
class TextSeqModel(MLModel):
    def __init__(self) -> None:
        super().__init__()
        self.model=load_model('lstm_trained_on_100_percent_lr=0.01_bs=128.keras')
    
    def predict(self, X):
        X_=np.zeros((len(X),len(X[0])))
        for i in range(len(X)):
            for j in range(len(X[0])):
                X_[i][j]=X[i][j]
        p=self.model.predict(X_)
        return (p>0.5).astype(int)
    
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == '__main__':
    # read datasets
    test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()
    
    # your trained models 
    text_model = TextSeqModel()
    
    # predictions from your trained models
    pred_text = text_model.predict(test_seq_X)
    
    # saving prediction to text files
    save_predictions_to_file(pred_text, "pred_text.txt")
