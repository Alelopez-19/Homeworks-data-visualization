import streamlit as st
import pandas as pd
import plotly.express as px
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

df = pd.read_csv("fmi_weather_and_price.csv", parse_dates = ['Time'])
df = df[df['Time'].dt.year == 2020]


figx = make_subplots(specs = [[{"secondary_y": True}]])

stddevp = df["Price"].std()
stddevt = df["Temp"].std()





figx = make_subplots(specs=[[{"secondary_y": True}]])


figx.add_trace(
    go.Scatter(x=df["Time"], y=df["Price"], mode="lines", name="Price"),
    secondary_y=False,
)


figx.add_trace(
    go.Scatter(x=df["Time"], y=df["Temp"], mode="lines", name="Temperature"),
    secondary_y=True,
)



figx.update_xaxes(title_text="Time")
figx.update_yaxes(title_text="Price", secondary_y=False)
figx.update_yaxes(title_text="Temperature", secondary_y=True)


figx.update_layout(title="Price and Temperature Year 2020")


st.plotly_chart(figx)

df_daily = df.resample('D', on='Time').mean()
df_daily.reset_index(inplace=True)




figx = make_subplots(specs=[[{"secondary_y": True}]])


figx.add_trace(
    go.Scatter(x=df_daily["Time"], y=df_daily["Price"], mode="lines", name="Price"),
    secondary_y=False,
)


figx.add_trace(
    go.Scatter(x=df_daily["Time"], y=df_daily["Temp"], mode="lines", name="Temperature"),
    secondary_y=True,
)


figx.update_xaxes(title_text="Time")
figx.update_yaxes(title_text="Price", secondary_y=False)
figx.update_yaxes(title_text="Temperature", secondary_y=True)


figx.update_layout(title="Price and Temperature 2020")


st.plotly_chart(figx)





df["Time"] = pd.to_datetime(df["Time"])


df_dailystd = df.resample('D', on='Time').var()
df_dailystd.reset_index(inplace=True)

figx = make_subplots(specs=[[{"secondary_y": True}]])


figx.add_trace(
    go.Scatter(x=df_dailystd["Time"], y=df_dailystd["Price"], mode="lines", name="Price"),
    secondary_y=False,
)


figx.add_trace(
    go.Scatter(x=df_dailystd["Time"], y=df_dailystd["Temp"], mode="lines", name="Temperature"),
    secondary_y=True,
)


figx.update_xaxes(title_text="Time")
figx.update_yaxes(title_text="Price Variance", secondary_y=False)
figx.update_yaxes(title_text="Temperature Variance", secondary_y=True)


figx.update_layout(title="Variance of Daily Price and Temperature (2020)")


st.plotly_chart(figx)






df_dailystd = df.resample('D', on='Time').std()
df_dailystd.reset_index(inplace=True)

figx = make_subplots(specs=[[{"secondary_y": True}]])


figx.add_trace(
    go.Scatter(x=df_dailystd["Time"], y=df_dailystd["Price"], mode="lines", name="Price"),
    secondary_y=False,
)


figx.add_trace(
    go.Scatter(x=df_dailystd["Time"], y=df_dailystd["Temp"], mode="lines", name="Temperature"),
    secondary_y=True,
)



figx.update_xaxes(title_text="Time")
figx.update_yaxes(title_text="Price Standard Deviation", secondary_y=False)
figx.update_yaxes(title_text="Temperature Standard Deviation", secondary_y=True)


figx.update_layout(title="Standard deviation of daily Price and Temperature  2020")


st.plotly_chart(figx)

df_dailymedian = df.resample('D', on='Time').median().reset_index()

fig_median = px.line(df_dailymedian, x="Time", y=["Price", "Temp"], 
                      title="Daily Median of Price and Temperature (2020)")

st.plotly_chart(fig_median)





df_dailymin = df.resample('D', on='Time').min().reset_index()
df_dailymax = df.resample('D', on='Time').max().reset_index()

fig_range = go.Figure()

fig_range.add_trace(go.Scatter(x=df_dailymin["Time"], y=df_dailymin["Price"], 
                               mode="lines", name="Min Price"))
fig_range.add_trace(go.Scatter(x=df_dailymax["Time"], y=df_dailymax["Price"], 
                               mode="lines", name="Max Price"))
fig_range.add_trace(go.Scatter(x=df_dailymin["Time"], y=df_dailymin["Temp"], 
                               mode="lines", name="Min Temp", yaxis="y2"))
fig_range.add_trace(go.Scatter(x=df_dailymax["Time"], y=df_dailymax["Temp"], 
                               mode="lines", name="Max Temp", yaxis="y2"))

fig_range.update_layout(title="Daily Min & Max of Price and Temperature (2020)",
                        yaxis2=dict(overlaying="y", side="right"))

st.plotly_chart(fig_range)



df["Month"] = df["Time"].dt.month

fig_box = px.box(df, x="Month", y="Price", title="Monthly Price Distribution (2020)")
st.plotly_chart(fig_box)

fig_box2 = px.box(df, x="Month", y="Temp", title="Monthly Temperature Distribution (2020)")
st.plotly_chart(fig_box2)



corr_matrix = df[["Price", "Temp", "Wind"]].corr()

fig_heatmap = plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
st.pyplot(fig_heatmap)


X = df[["Temp", "Wind"]]   
y = df["Price"]           


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)


st.write(f"Model Coefficients: {model.coef_}")
st.write(f"Intercept: {model.intercept_}")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")


fig_regression = px.scatter(x=y_test, y=y_pred, labels={'x': "Actual Price", 'y': "Predicted Price"},
                            title="Actual vs. Predicted Electricity Prices")
st.plotly_chart(fig_regression)


poly_degree = 2  


poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
X_poly = poly.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)


poly_model = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
poly_model.fit(X_train, y_train)


y_poly_pred = poly_model.predict(X_test)


mse_poly = mean_squared_error(y_test, y_poly_pred)
r2_poly = poly_model.score(X_test, y_test)


st.write(f"Polynomial Regression (Degree {poly_degree})")
st.write(f"Mean Squared Error: {mse_poly:.2f}")
st.write(f"R² Score: {r2_poly:.2f}")


fig_poly = px.scatter(x=y_test, y=y_poly_pred, labels={'x': "Actual Price", 'y': "Predicted Price"},
                      title=f"Actual vs. Predicted Prices (Polynomial Regression, Degree {poly_degree})")
st.plotly_chart(fig_poly)

st.write("Based on the analysis, it was found that weather conditions like "
"temperature and wind speed might account for only 20 percent of the variation in electricity prices, "
"this might mean a weak relationship between these variables and electricity price fluctuations. "

)
