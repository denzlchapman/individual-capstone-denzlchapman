
**Student Name:** Denz'l Chapman

**Dataset:** Industry Market Cap Dataset
https://www.kaggle.com/datasets/zibranzarif/industry-market-cap-analysis-dataset

**Problem Statement:** The objective of my project was to build machine learning models capable of predicting a company's market cap given the fundamental financial features these models were trained on. While analyzing features such as revenue, earnings, total assets, and other financial ratios, the goal was to understand how these features impact a company's value and if these models could make strong predictions using these features. Predicting a company's market cap given the fundamentals is extremely important for value investors and helps minimize speculation when investing. 

**Target Variable:** MarketCap

**Selected Features:** Revenue, Earning, Net Assets, Total Assets, Total Debt, Return on Assets, Return on Equity, Debt to Equity, Industry

**Best Regression Model:** Gradient Boosting Regressor, R2 Score: 0.8023

**Best Classification Model:** Logstic Regression, Accuracy 0.7517

**Deployed App URL:** https://individual-capstone-denzlchapman.streamlit.app

### Project Highlights

[Write 2-3 bullet points about interesting findings or challenges you overcame. This helps employers understand your thought process!]

- Seeing the difference in model performance during regression when including PE&PS ratio given the TARGET is Marketcap(Data Leakage).
- My original binning strategy wouldnt work because the dataset did not have enough data points to represent both MicroCap and MegaCap classifications.(less than 4% for each)  
- Precisely predicting a company's market cap value is extremely difficult because there are so many other influences than just the fundamentals alone. 
