from abc import ABC, abstractmethod
from itertools import takewhile
import datetime as dt
import pandas as pd
from tqdm import tqdm
from instaloader import Instaloader, Profile, Post


class Client(ABC):
    @abstractmethod
    def set_targets(self, *targets: str) -> 'Client':
        raise NotImplementedError()

    @abstractmethod
    def set_period(self, days: int) -> 'Client':
        raise NotImplementedError()

    @abstractmethod
    def get_posts(self):
        raise NotImplementedError()


class InstagramClient(Client):
    def __init__(self):
        self.client = Instaloader()
        self.targets = ()
        self.period = 30
        self.timenow = dt.datetime.utcnow()
    
    def set_targets(self, *targets: str) -> 'Client':
        self.targets = targets
        return self

    def set_period(self, days: int) -> 'Client':
        self.period = days
        return self

    def get_posts(self) -> dict[list[dict]]:
        if self.targets == ():
            raise Exception("No targets set, call set_targets first")
        
        date_limit = self.timenow - dt.timedelta(self.period)
        extracted_posts = []
        for username in tqdm(self.targets):
            posts = Profile.from_username(self.client.context, username).get_posts()
            user_post_datas = [self.__extract_post_data(post) for post in takewhile(lambda post: post.date > date_limit or post.is_pinned, posts)]
            extracted_posts.extend(user_post_datas)

        return pd.DataFrame(extracted_posts)
    
    def __extract_post_data(self, post: Post) -> dict:
        if not isinstance(post, Post):
            raise Exception("Invalid post dtype")
        else:
            post_data = {
                "username": post.owner_username,
                "caption": post.caption if post.caption != None else "",
                "caption_hashtags": post.caption_hashtags,
                "caption_mentions": post.caption_mentions,
                "likes": post.likes,
                "comments": post.comments,
                "location": post.location, 
                "date_utc": post.date_utc.strftime("%Y%m%d %H:%M:%S UTC"),
                "sponsors": post.sponsor_users,
                "tagged_users": post.tagged_users,
                "post_url": f"instagram.com/p/{post.shortcode}/",
            }
            return post_data


class TikTokClient(Client):
    def __init__(self):
        pass

    def set_targets(self, *targets: str) -> 'Client':
        raise NotImplementedError()

    def set_period(self, days: int) -> 'Client':
        raise NotImplementedError()

    def get_posts(self):
        raise NotImplementedError()