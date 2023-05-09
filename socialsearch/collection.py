import pandas as pd
from config_loader import config
from client import InstagramClient, TikTokClient


class DataCollector:
    def __init__(self):
        self.client = InstagramClient()

    def download(self, pkl_path: str, xlsx_path: str):
        self.client.set_targets(*config.influencers).set_period(365)
        posts = self.client.get_posts()
        posts.to_pickle(pkl_path)
        posts.to_excel(xlsx_path)

    def merge(self, pkl_path: str, match_path: str, dataset_path: str):
        '''Merge labelled match_XX.csv with unlabelled pkl dataframe, and exports to datasets/'''

        posts = pd.read_pickle(pkl_path)
        truth = pd.read_csv(match_path)["Sponsored"]
        merged = pd.concat([posts, truth], axis=1)
        merged.to_pickle(f"datasets/{dataset_path}")


if __name__ == "__main__":
    collector = DataCollector()
    print(config.influencers)
