from flask import Flask, render_template, request, jsonify, send_file, url_for
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import json
import os
import uuid
from pywebpush import webpush, WebPushException
import logging
from concurrent.futures import ThreadPoolExecutor
from flask_caching import Cache
from datetime import datetime

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY")
VAPID_CLAIMS = {
    "sub": os.getenv("VAPID_CLAIMS_SUB")
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 로그 파일 저장 경로 설정
log_directory = os.path.join(os.getcwd(), "static", "logs")
os.makedirs(log_directory, exist_ok=True)  # 디렉토리가 없으면 생성
log_filename = os.path.join(log_directory, "app_logs.xlsx")

if os.path.exists(log_filename):
    log_df = pd.read_excel(log_filename)
else:
    log_df = pd.DataFrame(columns=["timestamp", "level", "message"])

def save_log_to_excel():
    global log_df
    log_df.to_excel(log_filename, index=False)

class ExcelHandler(logging.Handler):
    def emit(self, record):
        global log_df
        log_entry = self.format(record)
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')  # 타임스탬프 형식 변경
        level = record.levelname
        message = log_entry
        new_log_entry = pd.DataFrame([{"timestamp": timestamp, "level": level, "message": message}])
        log_df = pd.concat([log_df, new_log_entry], ignore_index=True)
        save_log_to_excel()

excel_handler = ExcelHandler()
excel_handler.setLevel(logging.INFO)
logger.addHandler(excel_handler)

# 요청 전후로 로깅 추가
@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.url} from {request.remote_addr}")

@app.after_request
def log_response_info(response):
    logger.info(f"Response: {response.status} for {request.method} {request.url} from {request.remote_addr}")
    return response

models = {}
columns = []
subscriptions = []

current_macro_data = pd.DataFrame({
    'KR CCSI': [98.4],
    'KR CN currency': [187.91],
    'KR EU currency': [1475.43],
    'KR GDI': [489992.3],
    'KR GDP': [509845.7],
    'KR GNI': [481438.1],
    'KR JP currency': [869.14],
    'KR US currency': [1364.5],
    'KR coal energy consumption': [6040],
    'KR construction orders': [11673491],
    'KR consumers prices': [91.432],
    'KR corruption perception index': [29.3],
    'KR crime count': [161152.917],
    'KR departments': [45694.5],
    'KR edu_expense': [200400],
    'KR employment rate': [60.1],
    'KR entry': [50579.5],
    'KR finedust': [341],
    'KR gini': [0.345],
    'KR graduate': [558039],
    'KR house prices': [70.48],
    'KR households': [20153476],
    'KR interest': [3.5],
    'KR news sentiment': [109.2],
    'KR pop_elderly_rate': [19.2],
    'KR pop_natural_increase_rate': [-2.6],
    'KR total energy consumption': [21573],
    'KR unemployment rate': [3.5],
    'us_consumer_prices': [313.207],
    'us_gdp': [28284.5],
    'us_inflation_expectations': [3.1],
    'us_interest_rate': [5.50],
    'us_nonfarm_employment': [175000],
    'us_unemployment_rate': [3.9],
    'world_oil_prices': [77.8]
})

feature_translations = {
    'KR CCSI': '소비자심리지수(CCSI)',
    'KR CN currency': '중국환율',
    'KR EU currency': '유럽환율',
    'KR GDI': '국내총소득(GDI)',
    'KR GDP': '국내총생산(GDP)',
    'KR GNI': '국민총소득(GNI)',
    'KR JP currency': '일본환율',
    'KR US currency': '미국환율',
    'KR coal energy consumption': '총 석탄소비량',
    'KR construction orders': '건설수주액',
    'KR consumers prices': '소비자물가지수(CPI)',
    'KR corruption perception index': '부패인식지수',
    'KR crime count': '범죄발생건수',
    'KR departments': '출국자 수',
    'KR edu_expense': '사교육 지출액',
    'KR employment rate': '고용률',
    'KR entry': '입국자 수',
    'KR finedust': '미세먼지 지수',
    'KR gini': '지니계수',
    'KR graduate': '고등교육 졸업자 수',
    'KR house prices': '주택매매가격지수',
    'KR households': '세대 수',
    'KR interest': '금리',
    'KR news sentiment': '뉴스심리지수(NSI)',
    'KR pop_elderly_rate': '고령화지수',
    'KR pop_natural_increase_rate': '자연인구증가율',
    'KR total energy consumption': '총 에너지 소비량',
    'KR unemployment rate': '실업률',
    'us_consumer_prices': '미국 소비자물가지수',
    'us_gdp': '미국 GDP',
    'us_inflation_expectations': '미국 기대인플레이션',
    'us_interest_rate': '미국 기준금리',
    'us_nonfarm_employment': '미국 비농업 고용자수',
    'us_unemployment_rate': '미국 실업률',
    'world_oil_prices': '국제유가지수(WTI)'
}



### 재난 시나리오 데이터 생성
def generate_scenario_data(initial_value, monthly_decline, volatility, periods=12):
    return [round(initial_value * (1 + monthly_decline + np.random.uniform(-volatility, volatility))**month, 2) for month in range(periods)]

# 시나리오 데이터 생성 파라미터
scenario_params = {
    'KR CCSI': (-0.05, 0.02),  # 5% monthly decline, 2% volatility
    'KR CN currency': (0.03, 0.01),  # 3% monthly increase, 1% volatility
    'KR EU currency': (0.03, 0.01),
    'KR GDI': (-0.04, 0.02),
    'KR GDP': (-0.04, 0.02),
    'KR GNI': (-0.04, 0.02),
    'KR JP currency': (0.03, 0.01),
    'KR US currency': (0.03, 0.01),
    'KR coal energy consumption': (-0.03, 0.02),
    'KR construction orders': (-0.05, 0.02),
    'KR consumers prices': (0.02, 0.01),
    'KR corruption perception index': (0.05, 0.01),  # Increase in corruption perception
    'KR crime count': (0.04, 0.02),  # Increase in crime
    'KR departments': (-0.03, 0.02),
    'KR edu_expense': (-0.02, 0.01),
    'KR employment rate': (-0.03, 0.01),
    'KR entry': (-0.03, 0.02),
    'KR finedust': (0.01, 0.005),
    'KR gini': (0.01, 0.005),
    'KR graduate': (-0.02, 0.01),
    'KR house prices': (-0.04, 0.02),
    'KR households': (-0.01, 0.005),
    'KR interest': (0.01, 0.005),
    'KR news sentiment': (-0.05, 0.02),
    'KR pop_elderly_rate': (0.005, 0.002),
    'KR pop_natural_increase_rate': (-0.005, 0.002),
    'KR total energy consumption': (-0.02, 0.01),
    'KR unemployment rate': (0.02, 0.01),
    'us_consumer_prices': (0.01, 0.005),
    'us_gdp': (-0.01, 0.005),
    'us_inflation_expectations': (0.005, 0.002),
    'us_interest_rate': (0.005, 0.002),
    'us_nonfarm_employment': (-0.01, 0.005),
    'us_unemployment_rate': (0.01, 0.005),
    'world_oil_prices': (0.02, 0.01)
}

# 시나리오 데이터프레임 생성
scenario_data = {}
for key in current_macro_data.columns:
    initial_value = current_macro_data[key].iloc[0]
    monthly_decline, volatility = scenario_params[key]
    scenario_data[key] = generate_scenario_data(initial_value, monthly_decline, volatility)
disaster_scenario_df = pd.DataFrame(scenario_data)



# Covid-19 시나리오
real_macro_df = pd.read_excel('../Data/Summary/real_MacroEconomics.xlsx')
real_macro_df = real_macro_df.drop('Unnamed: 0', axis=1)
covid_scenario_df = real_macro_df[(real_macro_df['index'] == "2019/09") |
                                  (real_macro_df['index'] == "2019/10") |
                                  (real_macro_df['index'] == "2019/11") |
                                  (real_macro_df['index'] == "2019/12") | 
                                  (real_macro_df['index'] == "2020/01") | 
                                  (real_macro_df['index'] == "2020/02") | 
                                  (real_macro_df['index'] == "2020/03")]
covid_scenario_df = round(covid_scenario_df, 2)

@cache.cached(timeout=600, key_prefix='load_models')
def load_models():
    global models, columns
    bsi_columns = [col for col in bsi_data.columns if '업황실적' in col]
    models = {col: joblib.load(f'../model/xgboost_model_{col}.pkl') for col in bsi_columns}
    columns = bsi_columns

@cache.cached(timeout=600, key_prefix='load_data')
def load_data():
    global macro_data, bsi_data, current_bsi_values, bsi_latest_values, current_bsi_latest_values, merged_data, macro_columns, columns, X, Y
    macro_file = '../Data/Summary/MacroEconomic.xlsx'
    bsi_file = '../Data/Summary/Normalized_BSI_Data.xlsx'
    current_bsi_file = '../Data/산업/KR_bsi.xlsx'

    macro_data = pd.read_excel(macro_file)
    bsi_data = pd.read_excel(bsi_file)
    current_bsi_values = pd.read_excel(current_bsi_file)

    macro_data['index'] = pd.to_datetime(macro_data['index'], format='%Y/%m')
    bsi_data['업종_BSI'] = bsi_data['업종코드별'] + '_' + bsi_data['BSI코드별']
    bsi_data = bsi_data.drop(columns=['업종코드별', 'BSI코드별']).set_index('업종_BSI').transpose()
    bsi_data.index = pd.to_datetime(bsi_data.index, format='%Y/%m')

    current_bsi_values['업종_BSI'] = current_bsi_values['업종코드별'] + '_' + current_bsi_values['BSI코드별']
    current_bsi_values = current_bsi_values.drop(columns=['업종코드별', 'BSI코드별']).set_index('업종_BSI').transpose()
    current_bsi_values.index = pd.to_datetime(current_bsi_values.index, format='%Y/%m')
    
    columns = [col for col in current_bsi_values.columns if '업황실적' in col]

    bsi_latest_values = bsi_data[columns].iloc[-1].astype(float)
    current_bsi_latest_values = current_bsi_values[columns].iloc[-1].astype(float)

    merged_data = macro_data.merge(bsi_data, left_on='index', right_index=True, how='inner')

    macro_columns = macro_data.columns[1:]
    columns = [col for col in merged_data.columns if '업황실적' in col]

    X = merged_data[macro_columns]
    Y = merged_data[columns]

load_data()
load_models()

@cache.memoize(timeout=300)
def predict_scenario(scenario_data_scaled):
    with ThreadPoolExecutor() as executor:
        future_to_model = {executor.submit(models[col].predict, scenario_data_scaled): col for col in columns}
        predictions = {future_to_model[future]: future.result() for future in future_to_model}
    predictions_df = pd.DataFrame(predictions)
    return predictions_df

def send_web_push(subscription_information, message_body):
    return webpush(
        subscription_info=subscription_information,
        data=message_body,
        vapid_private_key=VAPID_PRIVATE_KEY,
        vapid_claims=VAPID_CLAIMS
    )


@app.route('/', methods=['GET'])
def home():
    features = current_macro_data.columns
    default_values = current_macro_data.iloc[0].to_dict()
    return render_template('index.html', features=features, default_values=default_values, feature_translations=feature_translations)


@app.route('/disaster', methods=['GET'])
def disaster():
    features = disaster_scenario_df.columns
    default_values = {k: ', '.join(map(str, v)) for k, v in disaster_scenario_df.to_dict(orient='list').items()}
    return render_template('index.html', features=features, default_values=default_values, feature_translations=feature_translations)

@app.route('/covid', methods=['GET'])
def covid():
    features = covid_scenario_df.columns
    default_values = {k: ', '.join(map(str, v)) for k, v in covid_scenario_df.to_dict(orient='list').items()}
    return render_template('index.html', features=features, default_values=default_values, feature_translations=feature_translations)

@app.route('/app', methods=['GET'])
def homeapp():
    features = current_macro_data.columns
    default_values = current_macro_data.iloc[0].to_dict()
    return jsonify({
        "features": features.tolist(),
        "default_values": default_values
    })

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json() or request.form.to_dict()

    for key in input_data:
        if input_data[key] == '':
            input_data[key] = current_macro_data[key].iloc[0]
    df = pd.DataFrame(input_data, index=[0])

    future_macro_data_list = []
    max_values = 0
    for column in df.columns:
        values = df[column]
        if isinstance(values[0], str):
            values = values[0].split(',')
        else:
            values = [values[0]]
        values = [float(value.strip()) for value in values if value.strip()]
        if values:
            max_values = max(max_values, len(values))
        future_macro_data_list.append(values)
        
    if max_values == 0:
        return jsonify({"error": "No valid input values provided."})

    for i in range(len(future_macro_data_list)):
        if len(future_macro_data_list[i]) < max_values:
            last_value = future_macro_data_list[i][-1]
            future_macro_data_list[i].extend([last_value] * (max_values - len(future_macro_data_list[i])))

    future_macro_data = pd.DataFrame(future_macro_data_list).T
    future_macro_data.columns = df.columns

    time_periods = pd.date_range(start=pd.to_datetime('now'), periods=max_values, freq='M')
    
    scenario_df = pd.concat([current_macro_data] * max_values, ignore_index=True)
    scenario_df['index'] = time_periods
    for column in future_macro_data.columns:
        scenario_df[column] = future_macro_data[column].values

    scenario_df = scenario_df.drop(columns=['index'])

    scaler = MinMaxScaler()
    scenario_scaled = scaler.fit_transform(scenario_df)

    try:
        predictions_df_scaled = predict_scenario(scenario_scaled)
    except ValueError as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)})

    predictions_df = predictions_df_scaled * (current_bsi_latest_values.max() - current_bsi_latest_values.min()) + current_bsi_latest_values.min()

    def calculate_diff_ratio(predictions_df, current_values):
        diff_ratio = (predictions_df - current_values) / current_values
        return diff_ratio

    diff_ratio = calculate_diff_ratio(predictions_df, current_bsi_latest_values)

    def create_heatmap(data, title, vmin, vmax, time_periods):
        heatmap = go.Figure(data=go.Heatmap(
            z=data.T.values,
            x=[period.strftime('%Y-%m') for period in time_periods],
            y=data.columns.str.replace("_업황실적", ""),
            colorscale='RdBu_r',
            zmin=vmin,
            zmax=vmax,
        ))
        heatmap.update_layout(
            title=title, 
            xaxis_title='Time Periods', 
            yaxis_title='BSI Columns', 
            xaxis=dict(tickformat='%Y-%m'),
            yaxis=dict(tickmode='array', tickvals=list(range(len(data.columns))), 
                       ticktext=data.columns.str.replace("_업황실적",""), automargin=True),
            height=800)
        return heatmap

    def create_line_graph(data, title, time_periods):
        line_graph = go.Figure()
        for col in data.columns:
            line_graph.add_trace(go.Scatter(
                x=[period.strftime('%Y-%m') for period in time_periods],
                y=data[col],
                mode='lines+markers',
                name=col.replace("_업황실적", "")
            ))
        line_graph.update_layout(
            title=title, 
            xaxis_title='Time Periods', 
            yaxis_title='Change Ratio', 
            height=800
        )
        return line_graph

    vmin = diff_ratio.min().min()
    vmax = diff_ratio.max().max()

    diff_heatmap = create_heatmap(diff_ratio, 'Change Ratio Heatmap', vmin, vmax, time_periods)
    line_graph = create_line_graph(diff_ratio, 'Change Ratio LineGraph', time_periods)

    diff_heatmap_data = json.loads(diff_heatmap.to_json())
    line_graph_data = json.loads(line_graph.to_json())

    excel_filename = f"prediction_results_{uuid.uuid4().hex}.xlsx"
    output_path = os.path.join(os.getcwd(), "static", excel_filename)
    
    with pd.ExcelWriter(output_path) as writer:
        scenario_df.to_excel(writer, sheet_name="Input Data")
        predictions_df.to_excel(writer, sheet_name="Predicted BSI Values")
        diff_ratio.to_excel(writer, sheet_name="Change Ratio")

    download_link = url_for('download_excel', filename=excel_filename)
    
    notification_payload = {
        "title": "SERA - Prediction Completed",
        "body": "Your prediction request has been processed.",
        "icon": "images/logo.png"
    }

    def send_notification(subscription):
        try:
            send_web_push(subscription, json.dumps(notification_payload))
        except WebPushException as ex:
            logger.error(f"Web push failed: {ex}")

    with ThreadPoolExecutor() as executor:
        executor.map(send_notification, subscriptions)
    
    return jsonify(diff_heatmap_data=diff_heatmap_data, line_graph_data=line_graph_data, download_link=download_link)

@app.route('/download_excel', methods=['GET'])
def download_excel():
    filename = request.args.get('filename')
    file_path = os.path.join(os.getcwd(), "static", filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)

@app.route('/subscribe', methods=['POST'])
def subscribe():
    subscription = request.get_json()
    subscriptions.append(subscription)
    return jsonify({"message": "Subscribed successfully."}), 201

@app.route('/send_notification', methods=['POST'])
def send_notification():
    message = request.get_json().get('message')
    subscription_info = request.get_json().get('subscription_info')
    
    try:
        send_web_push(subscription_info, message)
        return jsonify({"message": "Notification sent successfully"})
    except WebPushException as ex:
        logger.error(f"Notification failed: {ex}")
        return jsonify({"message": "Notification failed", "details": str(ex)}), 500

@app.route('/send_test_notification', methods=['GET'])
def send_test_notification():
    subscription_info = {
        # 실제 구독 정보로 대체해야 합니다
    }
    message_body = "This is a test notification from SERA"
    try:
        send_web_push(subscription_info, message_body)
        return jsonify({"success": True})
    except WebPushException as ex:
        logger.error(f"Web push failed: {ex}")
        if ex.response and ex.response.json():
            extra = ex.response.json()
            logger.error(f"Remote service replied with a {extra.code}:{extra.errno}, {extra.message}")
        return jsonify({"success": False}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
