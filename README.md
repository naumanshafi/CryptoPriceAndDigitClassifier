# CryptoPriceAndDigitClassifier

## Overview
This Streamlit application provides a dual functionality platform for users to track and compare cryptocurrency prices using the CoinGecko API, and to classify digit images using a TensorFlow model.

## Features

- **Cryptocurrency Performance Tracker**: Enables users to plot and view a cryptocurrency's price over the past year, along with the max and min price points during that timeframe.
  
- **Cryptocurrency Performance Comparison**: Allows users to compare the price performance of two selected cryptocurrencies over different time frames such as 1 week, 1 month, 1 year, and 5 years.

- **Digit Classification**: Provides an image classifier that accepts user-uploaded images of digits and classifies them using a pre-trained machine learning model.

## Installation

To set up the application locally, follow these steps:

\```bash
git clone https://github.com/A00475617/CryptoPriceAndDigitClassifier.git
cd CryptoPriceAndDigitClassifier
pip install -r requirements.txt
\```

## Usage

Execute the following command to launch the application:

\```bash
streamlit run app.py
\```

The app will be hosted locally at `http://localhost:8501`. Open this URL in a web browser to interact with the application.

## API Keys

The CoinGecko API requires an API key for requests. Ensure you have inserted your API key in the `app.py` script:

\```python
API_KEY = 'your_api_key_here'
\```

## Contributing

We welcome contributions and suggestions! Please feel free to fork the repository and submit pull requests.

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for more details.

## Project Link

- Repository: [https://github.com/A00475617/CryptoPriceAndDigitClassifier.git](https://github.com/A00475617/CryptoPriceAndDigitClassifier.git)
