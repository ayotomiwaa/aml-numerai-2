# Numerai Nexus - A Competitive Quest in Predicting Financial Trends

## Introduction
This project explores the application of Machine Learning (ML) to the financial market, revolutionizing trading and investment through data analysis and informed decision-making. ML unlocks insights from historical data, reveals patterns, and predicts future market movements. 
ML is a tool that enhances financial market understanding, informing investment decisions and optimizing returns. This project delves into the fusion of finance and technology, where data-driven strategies reshape trading and investment.

## Data
The Numerai dataset is a tabular dataset that describes the global stock market over time. Each row represents a specific stock at a particular point in time. Key elements include:
- Features: The dataset encompasses a wide range of quantitative attributes known about the stock at the time. These include fundamentals like P/E ratio, technical signals like RSI, market data like short interest, and secondary information. These features have been carefully made point-in-time to prevent data leakage issues.
- Era: Eras in the dataset represent distinct time periods, with feature values corresponding to that specific point in time, and target values looking forward from that point. It's recommended to treat each era as a single data point. In historical data (train and validation), eras are spaced one week apart.
- Target: The dataset's target represents stock market returns 20 days into the future and is designed to align with the hedge fund's strategy. The target can be interpreted as stock-specific returns that are not influenced by broader market trends, country or sector dynamics, or well-known factors.
  
## Installation
To run the project, follow these steps:

1. Clone the repository: `git clone <repository_url>`
2. Install Python 3.x
3. Install the required dependencies:
   - numerapi: `pip install numerapi`
   - pandas: `pip install pandas`
   - scikit-learn: `pip install scikit-learn`
   - lazypredict: `pip install lazypredict`
4. Run the main notebook: `Numerai EDA & Baseline.ipynb`

## Contributing
If you want to contribute to this project, please create a pull request with a detailed description of your changes.

## License
MIT

## Authors
- [Temitope Adeyeha](https://github.com/Adeyeha)
- [Uthman Jinadu](https://github.com/jinaduuthman)
- [Ezekiel Ayotomiwa](https://github.com/ayotomiwaa)
- [Hassan Mahmud]
