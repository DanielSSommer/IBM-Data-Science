import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Dados de exemplo: dosagem do medicamento (variável independente) e resposta (variável dependente)
dosagem = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
resposta = np.array([2.5, 3.5, 4.2, 5.0, 5.5, 6.0, 6.8, 7.2, 8.0, 8.5])

# Visualizar os dados
plt.scatter(dosagem, resposta, color='blue', label='Dados de Treinamento')
plt.title('Dosagem vs. Resposta')
plt.xlabel('Dosagem')
plt.ylabel('Resposta')
plt.legend()
plt.show()

# Transformar os dados para incluir termos polinomiais de ordem 2 (quadrático)
poly_features = PolynomialFeatures(degree=2)
dosagem_poly = poly_features.fit_transform(dosagem.reshape(-1, 1))

# Criar e ajustar o modelo de regressão polinomial
model = LinearRegression()
model.fit(dosagem_poly, resposta)

# Visualizar a linha de regressão polinomial
dosagem_fit = np.linspace(1, 10, 100)
dosagem_fit_poly = poly_features.transform(dosagem_fit.reshape(-1, 1))
resposta_fit = model.predict(dosagem_fit_poly)

plt.scatter(dosagem, resposta, color='blue', label='Dados de Treinamento')
plt.plot(dosagem_fit, resposta_fit, color='red', label='Regressão Polinomial (grau=2)')
plt.title('Dosagem vs. Resposta')
plt.xlabel('Dosagem')
plt.ylabel('Resposta')
plt.legend()
plt.show()

# Determinar a dosagem ótima para máxima eficácia
dosagem_otima = dosagem_fit[np.argmax(resposta_fit)]
resposta_maxima = np.max(resposta_fit)
print(f"Dosagem ótima para máxima eficácia: {dosagem_otima}")
print(f"Máxima eficácia esperada: {resposta_maxima}")