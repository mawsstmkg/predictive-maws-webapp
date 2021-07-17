import pandas as pd
import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler



# create sliding windows
def sliding_window(data, window=5):
	window += 1
	data_window = []
	for i in range(len(data)-window+1):
		sub_data = []
		for j in range(window):
			sub_data.append(data[i+j])
		data_window.append(sub_data)
	df = pd.DataFrame(data_window)
	df.columns = ['x'+str(i) for i in range(window-1)]+['x9']
	return df

def main():
	
	# Set page title, icon, layout wide (more used space in central area) and sidebar initial state
	st.set_page_config(page_title="Predictive MAWS STMKG", page_icon='logo.png', layout="wide", initial_sidebar_state="expanded")
	df = pd.read_excel('data_pros2.xlsx')
	
	# Central area header
	col1, col2, = st.beta_columns([1,9])
	col1.image("logo.png")
	col2.write("# Website Predictive Maintenance AWS STMKG")
	st.write('Selamat datang di website sistem predictive maintenance Automatic Weather Station (AWS).\
    	Predictive maintenance merupakan perawatan untuk mengantisipasi kegagalan sebelum terjadi kerusakan total pada alat.\
    	Predictive maintenance ini dilakukan pada komponen tertentu pada alat dengan melakukan analisa trend perilaku mesin pada alat.\
    	Adapun metode yang digunakan pada website ini untuk melakukan predictive maintenance adalah ARIMA dan RBF.\
    	Kedua metode ini menerapkan teknologi Machine Learning dalam menganalisis trend dalam rentang waktu tertentu. \
    	Pengguna dapat memilih satu diantara kedua metode tersebut untuk mendapatkan hasil prediksi.')
	
	selected_methods = st.sidebar.selectbox("Pilih metode",['ARIMA', 'RBF'])
	st.sidebar.markdown(
        """<hr style="height:1px;border:none;color:#fff;background-color:#999;margin-top:5px;margin-bottom:10px" /> """,
		unsafe_allow_html=True)
	
	files = st.sidebar.file_uploader(label='Upload data yang akan diprediksi',
                                     accept_multiple_files=False,
                                     type=['xlsx'])
									 
	# Allow example data loading when no custom data are loaded
	if not files:
		if st.sidebar.checkbox("Gunakan data contoh"):
			files = 'dataset.xlsx'
			
	try:
		data = pd.read_excel(files, engine="openpyxl")
		
		# Arah angin
		sc_winddir = StandardScaler()
		X_winddir = sliding_window(df['winddir'], 9)
		X_winddir = sc_winddir.fit_transform(X_winddir)
		df_winddir = sliding_window(data['winddir'], 9)
		if selected_methods == 'ARIMA':
			model_winddir = joblib.load('winddir_lin.pkl')
		else:
			model_winddir = joblib.load('winddir_rbf.pkl')
		winddir_pred = model_winddir.predict(sc_winddir.transform(df_winddir))[0]

		# Kecepatan angin
		sc_windspeed = StandardScaler()
		X_windspeed = sliding_window(df['windspeed'], 9)
		X_windspeed = sc_windspeed.fit_transform(X_windspeed)
		df_windspeed = sliding_window(data['windspeed'], 9)
		if selected_methods == 'ARIMA':
			model_windspeed = joblib.load('windspeed_lin.pkl')
		else:
			model_windspeed = joblib.load('windspeed_rbf.pkl')
		windspeed_pred = model_windspeed.predict(sc_windspeed.transform(df_windspeed))[0]

		# Kelembapan
		sc_rh = StandardScaler()
		X_rh = sliding_window(df['rh'], 9)
		X_rh = sc_rh.fit_transform(X_rh)
		df_rh = sliding_window(data['rh'], 9)
		if selected_methods == 'ARIMA':
			model_rh = joblib.load('rh_lin.pkl')
		else:
			model_rh = joblib.load('rh_rbf.pkl')
		rh_pred = model_rh.predict(sc_rh.transform(df_rh))[0]

		# Tekanan udara
		sc_press = StandardScaler()
		X_press = sliding_window(df['press'], 9)
		X_press = sc_press.fit_transform(X_press)
		df_press = sliding_window(data['press'], 9)
		if selected_methods == 'ARIMA':
			model_press = joblib.load('press_lin.pkl')
		else:
			model_press = joblib.load('press_rbf.pkl')
		press_pred = model_press.predict(sc_press.transform(df_press))[0]

		# Radiasi Matahari
		sc_solrad = StandardScaler()
		X_solrad = sliding_window(df['solrad'], 9)
		X_press = sc_solrad.fit_transform(X_solrad)
		df_solrad = sliding_window(data['solrad'], 9)
		if selected_methods == 'ARIMA':
			model_solrad = joblib.load('solrad_lin.pkl')
		else:
			model_solrad = joblib.load('solrad_rbf.pkl')
		solrad_pred = model_solrad.predict(sc_solrad.transform(df_solrad))[0]
		
		# Curah hujan
		sc_rain = StandardScaler()
		X_rain = sliding_window(df['rain'], 9)
		X_press = sc_rain.fit_transform(X_rain)
		df_rain = sliding_window(data['rain'], 9)
		if selected_methods == 'ARIMA':
			model_rain = joblib.load('rain_lin.pkl')
		else:
			model_rain = joblib.load('rain_rbf.pkl')
		rain_pred = model_rain.predict(sc_rain.transform(df_rain))[0]

		# Suhu
		sc_temp = StandardScaler()
		X_temp = sliding_window(df['temp'], 9)
		X_temp = sc_temp.fit_transform(X_temp)
		df_temp = sliding_window(data['temp'], 9)
		if selected_methods == 'ARIMA':
			model_temp = joblib.load('temp_lin.pkl')
		else:
			model_temp = joblib.load('temp_rbf.pkl')
		temp_pred = model_temp.predict(sc_temp.transform(df_temp))[0]

		st.write('''## Hasil Prediksi Kerusakan''')

		func = []

		winddir_cont = st.beta_expander(label="Arah Angin", expanded=False)
		with winddir_cont:
			st.write('### Arah Angin', round(winddir_pred, 3))
			if winddir_pred>360:
				st.write('#### Status')
				status='Over'
				st.markdown(f'<p style="background-color:#f94144;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
			elif winddir_pred>0:
				st.write('#### Status')
				status='Normal'
				st.markdown(f'<p style="background-color:#2e6f95;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
				func.append(1)
			else:
				st.write('#### Status')
				status='Warning'
				st.markdown(f'<p style="background-color:#f9c74f;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)

		windspeed_cont = st.beta_expander(label="Kecepatan Angin", expanded=False)
		with windspeed_cont:
			st.write('### Kecepatan Angin', round(windspeed_pred, 3))
			if windspeed_pred>180:
				st.write('#### Status')
				status='Over'
				st.markdown(f'<p style="background-color:#f94144;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
			elif windspeed_pred>0:
				st.write('#### Status')
				status='Normal'
				st.markdown(f'<p style="background-color:#2e6f95;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
				func.append(1)
			else:
				st.write('#### Status')
				status='Warning'
				st.markdown(f'<p style="background-color:#f9c74f;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
			
		rh_cont = st.beta_expander(label="Kelembapan", expanded=False)
		with rh_cont:
			st.write('### Kelembapan', round(rh_pred, 3))
			if rh_pred>100:
				st.write('#### Status')
				status='Over'
				st.markdown(f'<p style="background-color:#f94144;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
			elif rh_pred>0:
				st.write('#### Status')
				status='Normal'
				st.markdown(f'<p style="background-color:#2e6f95;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
				func.append(1)
			else:
				st.write('#### Status')
				status='Warning'
				st.markdown(f'<p style="background-color:#f9c74f;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
			
		press_cont = st.beta_expander(label="Tekanan Udara", expanded=False)
		with press_cont:
			st.write('### Tekanan Udara', round(press_pred, 3))
			if press_pred>1100:
				st.write('#### Status')
				status='Over'
				st.markdown(f'<p style="background-color:#f94144;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
			elif press_pred>800:
				st.write('#### Status')
				status='Normal'
				st.markdown(f'<p style="background-color:#2e6f95;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
				func.append(1)
			else:
				st.write('#### Status')
				status='Warning'
				st.markdown(f'<p style="background-color:#f9c74f;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
			
		solrad_cont = st.beta_expander(label="Radiasi Matahari", expanded=False)
		with solrad_cont:
			st.write('### Radiasi Matahari', round(solrad_pred, 3))
			if solrad_pred>1600:
				st.write('#### Status')
				status='Over'
				st.markdown(f'<p style="background-color:#f94144;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
			elif solrad_pred>0:
				st.write('#### Status')
				status='Normal'
				st.markdown(f'<p style="background-color:#2e6f95;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
				func.append(1)
			else:
				st.write('#### Status')
				status='Warning'
				st.markdown(f'<p style="background-color:#f9c74f;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)

		rain_cont = st.beta_expander(label="Curah Hujan", expanded=False)
		with rain_cont:
			st.write('### Curah Hujan', round(rain_pred, 3))
			if rain_pred>700:
				st.write('#### Status')
				status='Over'
				st.markdown(f'<p style="background-color:#f94144;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
			elif rain_pred>0:
				st.write('#### Status')
				status='Normal'
				st.markdown(f'<p style="background-color:#2e6f95;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
				func.append(1)
			else:
				st.write('#### Status')
				status='Warning'
				st.markdown(f'<p style="background-color:#f9c74f;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
				
		temp_cont = st.beta_expander(label="Temperatur", expanded=False)
		with temp_cont:
			st.write('### Temperatur', round(temp_pred, 3))
			if temp_pred>60:
				st.write('#### Status')
				status='Over'
				st.markdown(f'<p style="background-color:#f94144;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
			elif temp_pred>19.30:
				st.write('#### Status')
				status='Normal'
				st.markdown(f'<p style="background-color:#2e6f95;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
				func.append(1)
			else:
				st.write('#### Status')
				status='Warning'
				st.markdown(f'<p style="background-color:#f9c74f;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
				
		handal = len(func)/7*100
		st.write('#### Tingkat Kehandalan', round(handal, 3), '%')

		st.write('### Petunjuk')
		st.write('#### Status')
		st.write('Parameter status menunujukkan kondisi atau status dari suatu sensor')
		
		status='Normal'
		st.markdown(f'<p style="background-color:#2e6f95;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
		st.write('100-70%')
		status='Warning'
		st.markdown(f'<p style="background-color:#f9c74f;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
		st.write('70-50%')
		status='Over'
		st.markdown(f'<p style="background-color:#f94144;color:#333333;font-size:24px;border-radius:2%;">{status}</p>', unsafe_allow_html=True)
		st.write('< 50%')
	
	except:
		pass
	
if __name__ == "__main__":
	main()


