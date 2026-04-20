
# Fashion Review Recommendation Prediction ![Static Badge](https://img.shields.io/badge/version-1.0.0-green)

This repostiory contains code and resources to run predicitions on reviews for fashion items (items of clothing) left by customers to determine whether or not the item/product is recommended by the customer, based on the review, or not recommended.

This task is framed as:

Binary text classification (recommend vs not recommend)

Labels:

1 = recommends
0 = does not recommend

It is not pure sentiment analysis:

A review can be positive but not recommend (“Nice design but size runs too small”)
A review can be negative but still recommend (“Quality is average, still good for the price”)

✅ So we care about intent to recommend, not just polarity

Predictions are done using NLP. spaCy is the primiary library used for NLP and I have additionally created a 'Sentiment Score' feature to assist with prediction using HuggingFace.

Data pre-processing and predictions are implemented using Scikit-Learn Pipelines.


## Created
- Project created: April 2026
- Readme updated: 19 April 2026

## Project Structure

pipeline_project/
├── data/
│   └── reviews.csv                     # Primary dataset
├── piepline_cache/                     # (Note: check for spelling vs 'pipeline_cache')
├── pipeline_cache/                     # Joblib caching folder storing the cached output of pipeline steps
├── Fashion_prediction_pipeline.ipynb   # Main project notebook
├── LICENSE.txt                         # Project license info
├── README.md                           # Project documentation

## Important files in the repository

<code>Fashion_prediction_pipeline.ipynb</code> - this is the jupyter notebook that contains the EDA, data pre-processing and classification. This is the notebook to run
<code>data/reviews.csv</code> - the labelled data set containing the fashion reviews on which the model is trained and predictions are run.

## Dependencies
You will need <code>python</code> along with the 'standard' data science related libraries we all know and love to run the <code>notebook</code> as well as those required for the NLP and Sentiment Analysis

Libraries (latest versions will run fine)

- numpy
- pandas
- matplotlib
- seaborn
- sklearn (scikit learn)
- tqdm
- functools (from Python standard library)
- os (from Python standard library)

Libraries for NLP
- spacy with 'en_core_web_md' model

Libraries for Sentiment Analysis with HuggingFace
- torch (Pytorch)
- transformers
- HuggingFace model: DistilBERT SST-2 pretrained model -  'distilbert-base-uncased-finetuned-sst-2-english'


## Running the Notebook ▶️


#### Expected Runtime

To optimize performance, the pipeline utilizes Joblib Memory caching. This ensures that deterministic preprocessing steps (Lemmatization and Sentiment Scoring) are computed only once per cross-validation fold. This maintains a clear Inference Contract, as the pipeline accepts raw text and handles all transformations internally without requiring external offline scripts.

However the HuggingFace (sentiment score) feature generation and Lemmatization are computationally heavy and will therefore likely take sometime to run.
For a PC with an Intel Core Ultra 5 - 135U with 16 GB RAM the tasks take a collective ~13 minutes to complete on the Training data (80% of the total dataset)

## Credits 🤝
A huge thanks to the Udacity teams without whom this project would not have been possible.

## License 📜
As per License.txt file.

We ❤️ [Udacity!](https://udacity.com)

