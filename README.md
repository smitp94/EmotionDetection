# EmotionDetection
**Inspiration**
A successful business knows what its customers need and feel about their products. Therefore we developed a tool for generating Emotion Analysis about a business.

**How**
Using Long Short Term Memory (LSTM) we are training a model to classify emotional state of a person writing that review. Using those classifications we are providing analysis report about emotion distribution of that business

How we built it
*Model: *We used tensorflow library to train LSTM to classify emotional state. 
*Data Acquisition: *We used SemEval 2007 Task 14 for training data. 
*Web Framework: *Using Flask we created web application's architecture 
*Testing Data: *User review about a business were collected by scraping Yelp website. 
*Final Product: *Reviews were classified using trained LSTM model and report was generated using Matplotlib.
