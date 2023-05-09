import joblib
import pandas as pd
from config_loader import config
from client import InstagramClient

def main():
    # live test of current pickled model
    client = InstagramClient()
    influencers = ["kingjames"]
    client.set_targets(*influencers).set_period(days=10)
    posts = client.get_posts()
    
    model = joblib.load(config.models["model_path"])
    yhat = model.predict(posts)
    
    compare = pd.concat([posts["caption"], pd.Series(yhat)], axis=1)
    print(compare)

if __name__ == "__main__":
    main()