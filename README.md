# Машинное обучение в криптовалюте
*Это мой самостоятельный проект, который я решил сделать сам, так как тема криптовалюты для меня показалась очень интересной*

## Описание проекта
С помощью данных, взятых с криптовалютного рынка Binance, о пяти случайных и не слишком популярных монетах, попробуем создать модель, которая будет предсказывать рост или падение цены за последние пару часов на определенную монету. Это поможет определить и создать правила для точек входа и выхода из различных позиций

## Используемые библиотеки
import requests
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
