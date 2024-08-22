import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set the title and headers for the application
st.title('Cryptocurrency Performance and Digit Classification App')

tab1, tab2, tab3 = st.tabs(["Cryptocurrency Performance Tracker", "Cryptocurrency Performance Comparison", "Digit Classification"])

# API configuration
API_COINS_LIST_URL = 'https://api.coingecko.com/api/v3/coins/list'
API_COIN_MARKET_CHART_URL = 'https://api.coingecko.com/api/v3/coins/{id}/market_chart'
API_KEY = 'CG-TZe1VrxroETPEuLXxfBFTtEU'
headers = {
    'accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}


model_path = 'model.keras'
digit_recognizer = tf.keras.models.load_model(model_path)

def classify_image(original_img):
    if original_img.mode in ('RGBA', 'LA', 'P'):
        background_layer = Image.new("RGB", original_img.size, "WHITE")
        background_layer.paste(original_img, mask=original_img.getchannel('A') if original_img.mode == 'RGBA' else original_img)
        processed_img = background_layer
    else:
        processed_img = original_img.convert('RGB')

    processed_img = ImageOps.grayscale(processed_img)
    processed_img = ImageOps.invert(processed_img)

    aspect = min(28 / processed_img.width, 28 / processed_img.height)
    new_dimensions = (int(processed_img.width * aspect), int(processed_img.height * aspect))
    processed_img = processed_img.resize(new_dimensions, Image.Resampling.LANCZOS)

    canvas = Image.new('L', (28, 28), 'white')
    paste_coords = ((28 - new_dimensions[0]) // 2, (28 - new_dimensions[1]) // 2)
    canvas.paste(processed_img, paste_coords)

    img_array = np.array(canvas).astype(np.float32) / 255.0
    img_array = img_array.reshape(-1, 28, 28, 1) 

    predictions = digit_recognizer.predict(img_array)
    return np.argmax(predictions)

def fetch_coin_data(coin_id, days):
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(API_COIN_MARKET_CHART_URL.format(id=coin_id), params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['volume'] = pd.DataFrame(data['total_volumes'])[1]
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        return df
    else:
        st.error(f'Failed to fetch data for {coin_id}. Status Code: {response.status_code}')
        return pd.DataFrame()
    
# Fetch coins list with caching
@st.cache(ttl=86400)
def fetch_coins_list():
    for i in range(3):
        response = requests.get(API_COINS_LIST_URL, headers=headers)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        elif response.status_code == 429:
            time.sleep(2 ** i)
        else:
            st.error(f'Failed to fetch cryptocurrency list. Status Code: {response.status_code}')
            return pd.DataFrame()
    st.error('Failed to fetch after multiple attempts.')
    return pd.DataFrame()

coins_list_df = fetch_coins_list()

with tab1:
    st.header("Cryptocurrency Performance Tracker")
    # User input for cryptocurrency name
    cryptocurrency_name = st.text_input("Enter a cryptocurrency name", "bitcoin", key='tracker').lower().strip()

    if cryptocurrency_name in coins_list_df['id'].values:
        response = requests.get(API_COIN_MARKET_CHART_URL.format(id=cryptocurrency_name), params={'vs_currency': 'usd', 'days': '365'}, headers=headers)
        if response.ok:
            price_data = response.json().get('prices', [])
            if price_data:
                data = pd.DataFrame(price_data, columns=['timestamp', 'price'])
                data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('date', inplace=True)
                st.line_chart(data['price'])

                max_price = data['price'].max()
                min_price = data['price'].min()
                max_date = data[data['price'] == max_price].index.strftime('%Y-%m-%d')[0]
                min_date = data[data['price'] == min_price].index.strftime('%Y-%m-%d')[0]
                st.write(f"Maximum price: ${max_price:.2f} on {max_date}")
                st.write(f"Minimum price: ${min_price:.2f} on {min_date}")
            else:
                st.error("No price data available for this cryptocurrency.")
        else:
            st.error("Failed to fetch data for the selected cryptocurrency.")
    else:
        st.error(f"The cryptocurrency named '{cryptocurrency_name}' is not recognized.")

with tab2:
    st.header("Cryptocurrency Performance Comparison")
    # Input fields for cryptocurrency names
    first_coin = st.text_input("Enter the first cryptocurrency name", "bitcoin", key='comparison1').lower().strip()
    second_coin = st.text_input("Enter the second cryptocurrency name", "ethereum", key='comparison2').lower().strip()

    # Dropdown for selecting the time frame
    time_frame = st.selectbox("Select time frame", options=["7", "30", "365", "1825"],
                              format_func=lambda x: f"{int(x)//365} year" if int(x) > 30 else f"{x} days", key='timeframe')

    if first_coin in coins_list_df['id'].values and second_coin in coins_list_df['id'].values:
        first_coin_data = fetch_coin_data(first_coin, time_frame)
        second_coin_data = fetch_coin_data(second_coin, time_frame)

        if not first_coin_data.empty and not second_coin_data.empty:
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(x=first_coin_data.index, y=first_coin_data['price'], mode='lines', name=f'{first_coin.capitalize()} Price'))
            price_fig.add_trace(go.Scatter(x=second_coin_data.index, y=second_coin_data['price'], mode='lines', name=f'{second_coin.capitalize()} Price'))
            price_fig.update_layout(title='Price Comparison', yaxis_title='Price in USD', xaxis_title='Date')
            st.plotly_chart(price_fig)

            volume_fig = go.Figure()
            volume_fig.add_trace(go.Scatter(x=first_coin_data.index, y=first_coin_data['volume'], mode='lines', name=f'{first_coin.capitalize()} Volume', line=dict(color='green')))
            volume_fig.add_trace(go.Scatter(x=second_coin_data.index, y=second_coin_data['volume'], mode='lines', name=f'{second_coin.capitalize()} Volume', line=dict(color='orange')))
            volume_fig.update_layout(title='Volume Comparison', yaxis_title='Volume', xaxis_title='Date')
            st.plotly_chart(volume_fig)
            
with tab3:
    st.header("Digit Classification App")
    st.write("Upload an image of a digit, and the system will determine which digit it is.")
    uploaded_image = st.file_uploader("Select an image...", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Your Image', use_column_width=True)
        prediction_result = classify_image(image)
        st.write(f'The predicted digit is: {prediction_result}')