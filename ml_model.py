import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

dataset = pd.read_csv("dados-regressao-logistica-tratados.csv", delimiter=",")

x_train, x_test, y_train, y_test = train_test_split(dataset.drop('AVC', axis=1),
                                                    dataset['AVC'],
                                                    test_size=0.30,
                                                    random_state=101)

log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
log_model.fit(x_train, y_train)

def testar_ia():
  print("Descubra se você tem chances de ter AVC: ")
  idade = int(input("\nQual a sua idade?: "))
  genero_escolha = int(input("\nDigite seu gênero sendo 0 feminino e 1 masculino: "))
  regiao = int(input("\nDigite 0 se você mora na região urbana e 1 se mora na rural: "))
  condicao_fumante = int(input("\nInsira 0 se você nunca fumou, 1 se você é fumante e 2 se você é ex-fumante: "))
  trabalho = int(input("\nInsira seu tipo de trabalho sendo 0 para nunca trabalhou, 1 para mercado privado,\n2 para autônomo e 3 para funcionário público. Caso tenha trabalhado em mais de um\ntipo de trabalho ao longo de sua vida, insire o tipo no qual você\npassou mais tempo. Se for criança, digite 4: "))
  estado_civil = int(input("\nVocê é ou já foi casado alguma vez? 0 para não e 1 para sim: "))
  doenca_cardiaca = int(input("\nTem alguma doença cardíaca? 0 para não e 1 para sim: "))
  hipertensao = int(input("\nTem hipertensão? 0 para não e 1 para sim: "))
  glicose = float(input("\nQual seu nível médio de glicose? Caso não saiba, digite apenas 0: "))
  peso = float(input("\nDigite seu peso em kg: "))
  altura = float(input("\nDigite sua altura em metros: "))

  feminino = 1 if genero_escolha == 0 else 0
  masculino = 1 if genero_escolha == 1 else 0

  nunca_fumou = 1 if condicao_fumante == 0 else 0
  fumante = 1 if condicao_fumante == 1 else 0
  ex_fumante = 1 if condicao_fumante == 2 else 0

  nunca_trabalhou = 1 if trabalho == 0 else 0
  privado = 1 if trabalho == 1 else 0
  autonomo = 1 if trabalho == 2 else 0
  governo = 1 if trabalho == 3 else 0
  crianca = 1 if trabalho == 4 else 0

  rural = 1 if regiao == 0 else 0
  urbano = 1 if regiao == 1 else 0

  glicose_final = dataset['Nivel medio de glicose'].mean() if glicose == 0 else glicose
  glicose_format = round(glicose_final, 2)

  imc = peso / (altura*altura)
  imc_format = round(imc, 1)

  exemplo = np.array([idade,
                      hipertensao,
                      doenca_cardiaca,
                      estado_civil,
                      glicose_format,
                      imc_format,
                      feminino,
                      masculino,
                      governo,
                      nunca_trabalhou,
                      privado,
                      autonomo,
                      crianca,
                      rural,
                      urbano,
                      ex_fumante,
                      nunca_fumou,
                      fumante]).reshape((1,-1))

  resultado = int(log_model.predict(exemplo)[0])

  print("\nCalculando...")

  if resultado == 0:
    print("\nVocê tem baixas chances de ter um AVC no futuro")
  else:
    print("\nVocê tem alta chance de ter um AVC no futuro")

testar_ia()