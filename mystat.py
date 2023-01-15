import pandas as pd
import numpy as np
import matplotlib.pyplot as pypl
import rpy2.robjects.numpy2ri
#from rpy2.robjects.packages import importr
#rpy2.robjects.numpy2ri.activate()
#r_stats = importr('stats')

from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson

from scipy.stats import (
        contingency,    # contingency.expected_freq
        chi2_contingency,   
        fisher_exact,
        f_oneway,       # ANOVA
        levene,         # гетероскедастичность чееек
        shapiro
)

def load_data() -> pd.DataFrame:
    print('Загружаем данные...')
    coders_data_big = pd.read_excel(io="2016-FCC-New-Coders-Survey-Data.xlsx", sheet_name=0)

    columns = [
            'EmploymentField', 'EmploymentStatus', 'Gender', 'LanguageAtHome', 
            'JobWherePref', 'SchoolDegree', 'Income',
            'CityPopulation', 'HasDebt', 'JobPref', 'MaritalStatus'
            ]
    
    coders_data = coders_data_big[columns]

    for c in columns:
        coders_data = coders_data[coders_data[c].notna()]
    coders_data = coders_data[coders_data['Gender'].isin(('male', 'female'))]

    # стандартизируем колонку Income
    #coders_data['Income'] = (coders_data['Income'] - coders_data['Income'].mean()) / coders_data['Income'].std()
    print('Успешно')

    return coders_data

def add_labels(arr : np.ndarray, c, r):
    arr = np.vstack((c, arr))
    arr = np.column_stack((r, arr))

    return arr

def create_tables(data : pd.DataFrame, field1 : str, field2 : str):
    columns = data[field1].unique().astype(str).tolist() + ["ALL"]
    rows = [field1 + field2] + data[field2].unique().astype(str).tolist() + ["ALL"]
    link_table = pd.crosstab(data[field1], data[field2], margins=True)
    exp = np.transpose(np.round(contingency.expected_freq(link_table), 2))

    # представление таблицы link_table в виде двумерной матрицы из значений (для отображения в HTML)
    lt = np.transpose(link_table.to_numpy())
    lt = add_labels(lt, columns, rows)

    exp = add_labels(exp, columns, rows)
    #print(lt)

    return (lt, exp)

def choose_method(data : pd.DataFrame, field1 : str, field2 : str):
    '''
    При анализе четырехпольных таблиц ожидаемые значения в каждой из ячеек должны
    быть не менее 10 
    В том случае, если хотя бы в одной ячейке ожидаемое явление
    принимает значение от 5 до 9 критерий хи квадрат должен рассчитываться с
    поправкой Йейтса 
    Если хотя бы в одной ячейке ожидаемое явление меньше 5 то для
    анализа должен использоваться точный критерий Фишера 
    В случае анализа многопольных таблиц ожидаемое число наблюдений не должно принимать значения
    менее 5 более чем в 20 ячеек, иначе используем Тест Фримана Холтона
    '''

    ct = pd.crosstab(data[field1], data[field2], margins=True).to_numpy()
    grades = {
        ">10" : 0,
        "5-10" : 0,
        "<5" : 0
    }

    for ct_row in ct:
        for cell in ct_row:
            if float(cell) > 10:
                grades['>10'] += 1
            elif 5 <= float(cell) <= 10:
                grades['5-10'] += 1
            else:
                grades['<5'] += 1
    
    if grades['<5'] > 0:
        if grades['<5'] < 20:
            return "phisher"
        else:
            return "freeman" 
    elif grades['5-10'] > 0:
        return "corrected chi2"
    else:
        return "chi2"

def perform_test(cross_table : pd.DataFrame, method):
    # тест Фишера не работает для таблиц сопряженнности шире чем 2х2
    # тест Фишера из R не подключается
    # кринж, что тут сказать
        #return fisher_exact(cross_table)
        #return r_stats.fisher_test(np.array(cross_table.values()))
    
    if method == 'chi2':
        test = chi2_contingency(cross_table, correction=False)
    else:
        test = chi2_contingency(cross_table, correction=True)

    fstat, p = test[0], test[1]
    if p > 0.05:
        itr = "p-value > 0.05, столбцы зависимые (значения одной выборки неравномерно распределены среди значений другой выборки)"
    else:
        itr = "p-value < 0.05, столбцы независимые (значения одной выборки равномерно распределены среди значений другой выборки)"
    
    return (fstat, p, itr)

def check_normality(column : pd.Series):
    # проверяем по тесту Шапиро-Уилка нормальность обычной выборки
    shapiro_p = shapiro(column.values)[1]

    if shapiro_p > 0.05:
        return (shapiro_p)
    
    # лог-трансформация
    column_log = column.apply(lambda x : np.log(x + 1))
    shapiro_log_p = shapiro(column_log.values)[1]
    
    if shapiro_log_p > 0.05:
        return (shapiro_p, shapiro_log_p)
    
    # урезать выборку до 100 наблюдений
    column_cut = column.sample(100, ignore_index=True)
    shapiro_cut_p = shapiro(column_cut.values)[1]

    if shapiro_cut_p > 0.05:
        return (shapiro_p, shapiro_log_p, shapiro_cut_p, True)
    else:
    # если никакой из вариантов не прошёл, значит, выборка не принадлежит нормальному распределению
        return (shapiro_p, shapiro_log_p, shapiro_cut_p, False)

def anova(data : pd.DataFrame, field1 : str, field2 : str):
    norm = check_normality(data[field1])
    if len(norm) == 2:
        data[field1] = data[field1].apply(lambda x : np.log(x + 1))
    elif len(norm) == 4:
        data = data.sample(100, ignore_index=True)

    # создаём группы наблюдений - зависимый признак с различными значениями независимого
    groups = list()
    for u in data[field2].unique():
        groups.append(data[data[field2] == u][field1])

    f_anova = f_oneway(*groups)[1]
    hsk = levene(*groups)[1]

    model = ols(f'{field1} ~ {field2}', data).fit()
    dw = durbin_watson(model.resid)

    str_table_ = str(model.summary().tables[2])
    str_table_ = str_table_.replace('=', '', -1)
    str_table_ = str_table_.replace(': ', ':', -1)
    str_table_ = str_table_.replace('Cond. No. ', 'Cond.No.:')
    str_table_ = str_table_.replace(' (JB)', '')
    list_table_ = str_table_.split()

    additional_summary = [list_table_[:4], list_table_[4:8], list_table_[8:12], list_table_[12:]]

    return (f_anova, hsk, dw, additional_summary)

    

    