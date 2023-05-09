# socialsearch

## Instagram Public Search + Sponsored Post Classifier
- This repository contains an end-to-end classifier built on XGBoost to detect sponsored posts on Instagram.
- The Instagram public account scraper is built on top of <a href="https://github.com/instaloader/instaloader">Instaloader</a>

## Note
- You have to do your own data labelling, since sponsored posts may not be explicitly declared on Instagram (which means that the IG Graph API will not tag it as a sponsored post)
  - This is the motivation for building a classifier to detect sponsored posts by influencers.

## Setup
The pipeline is controlled via config.yaml. 
```bash
pip install -r requirements.txt
```

## Performance
- Sponsored post detection is not a difficult problem to solve with ML: The XGBoost model is trained on SBert embeddings of Instagram posts' captions, as well as some other engineered features, such as caption length and number of tagged accounts. 
  - Some useful papers on arxiv: 
    - https://arxiv.org/abs/2011.05757
    - https://arxiv.org/abs/2111.03916
- I get about 90% out-of-sample performance on accuracy, recall and f1 metrics.

## Usage
- WIP

## Contributing
- Feel free to contribute: I am looking for a way to obtain data from other social media platforms as well (mainly TikTok). If you have a solution, you can build it on top of the abstract base class in `client.py`.
- Please take note of the `GNU GPLv3` license.