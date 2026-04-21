
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

Predictions are done using NLP. spaCy is the primiary library used for NLP.

Data pre-processing and predictions are implemented using Scikit-Learn Pipelines.


## Created
- Project created: April 2026
- Readme updated: 21 April 2026

## Project Structure

pipeline_project/
├── data/
│   └── reviews.csv                             # Primary dataset
├── Fashion_prediction_pipeline.ipynb           # Main project notebook
├── final_Fashion_Classification_mdoel.pkl      # export of the final tuned model for production use 
├── LICENSE.txt                                 # Project license info
├── nlp_stages.py                               # Separate file containing Spacy Lemmatizer custom transformer class. Done in this manner to keep memory cache file paths shorter. 
├── README.md                                   # Project documentation


## Important files in the repository

<code>Fashion_prediction_pipeline.ipynb</code> - this is the jupyter notebook that contains the EDA, data pre-processing and classification. This is the notebook to run
<code>data/reviews.csv</code> - the labelled data set containing the fashion reviews on which the model is trained and predictions are run.
<code>nlp_stages.py</code> - Separate file containing Spacy Lemmatizer custom transformer class. Done in this manner to keep memory cache file paths shorter.

## Dependencies
You will need <code>python</code> along with the 'standard' data science related libraries we all know and love to run the <code>notebook</code> as well as those required for the NLP

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
- spacy with 'en_core_web_sm' model


## Running the Notebook ▶️

The first <code>code</code> cell in the notebook must be run. 
Thereafter all cells from the 'Data Preparation' section must be run. Alternatively run all cells starting at the beggining.

Ensure the spaCy library is installed with the 'en_core_web_sm' model.


Steps to install spacy with en_core_web_sm:

pip install -U spacy
python -m spacy download en_core_web_sm

#### Expected Runtime: ~ 25 mins (on an intel core ultra 5 with 16 GB ram)

NOTE: To optimize performance, the pipeline utilizes Joblib Memory caching within the Lemmatiser transformer. This ensures that deterministic preprocessing steps (Lemmatization) are computed only once per cross-validation fold. This maintains a clear Inference Contract, as the pipeline accepts raw text and handles all transformations internally without requiring external offline scripts.

## Credits 🤝
A huge thanks to the Udacity teams without whom this project would not have been possible.

## License 📜
As per License.txt file.

We ❤️ [Udacity!](https://udacity.com)

