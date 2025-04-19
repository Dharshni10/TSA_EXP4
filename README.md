# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

### AIM:
To implement ARMA model in python.

### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data=pd.read_csv('gold.csv')

N = 1000
plt.rcParams['figure.figsize'] = [12, 6]
X = data['USD (AM)'][::-1].reset_index(drop=True)
plt.plot(X)
plt.title('gold Prices (USD)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Gold Prices ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('gold Prices PACF')

plt.tight_layout()
plt.show()
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])  
ma2 = np.array([1, theta1_arma22, theta2_arma22])  
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()

```
### OUTPUT:

#### Original data:

![Original](https://github.com/user-attachments/assets/0690e346-a316-443a-aa23-fbcb2525fa71)

#### Partial Autocorrelation

![Partial](https://github.com/user-attachments/assets/8270ae5e-588f-42e7-9cf9-e5300fac1e62)

#### Autocorrelation

![Autocorrelation](https://github.com/user-attachments/assets/00a7ea4a-e2e2-439b-9fd4-bb0877186a42)

#### SIMULATED ARMA(1,1) PROCESS:

![Simulated](https://github.com/user-attachments/assets/3d82fe28-1207-4efa-b237-618e0d2abea9)

#### Partial Autocorrelation

![Partial](https://github.com/user-attachments/assets/0521f648-d98c-47e4-a85b-4e6e731a18fe)

#### Autocorrelation

![Autocorrelation](https://github.com/user-attachments/assets/c5e742d8-aebb-4cd6-9877-29eb3bb6e6d2)

#### SIMULATED ARMA(2,2) PROCESS:

![Simulated](https://github.com/user-attachments/assets/3cb0f064-6cc0-4705-994e-e06558cf36d7)

#### Partial Autocorrelation

![Partial](https://github.com/user-attachments/assets/29cbbb8d-ee23-4216-9bcb-9a5c5ce00ce4)

#### Autocorrelation

![Autocorrelation](https://github.com/user-attachments/assets/9d70f07f-118e-4e52-970d-fd691b390fdd)

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
