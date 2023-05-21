import pandas as pd
import numpy as np
import psycopg2

from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson

from scipy.stats import (
    contingency,
    chi2_contingency,
    f_oneway,  # ANOVA
    levene,
    shapiro,
)


def connect_execute_db(query: str, args=None):
    connection = psycopg2.connect(
        host="localhost",
        port=5432,
        database="flaskdb",
        user="postgres",
        password="postgredb1",
    )

    cursor = connection.cursor()

    result = None
    cursor.execute(query, args)
    connection.commit()
    try:
        result = pd.DataFrame(cursor.fetchall())
    except psycopg2.ProgrammingError:
        pass

    cursor.close()
    connection.close()

    return result


def load_data() -> pd.DataFrame:
    coders_data = connect_execute_db("SELECT * FROM flaskdb;")

    coders_data.columns = [
        "CityPopulation",
        "EmploymentField",
        "EmploymentStatus",
        "Gender",
        "HasDebt",
        "Income",
        "JobPref",
        "JobWherePref",
        "LanguageAtHome",
        "MaritalStatus",
        "SchoolDegree",
    ]

    return coders_data


def add_labels(arr: np.ndarray, to_columns, to_rows):
    arr = np.vstack((to_columns, arr))
    arr = np.column_stack((to_rows, arr))

    return arr


def create_tables(data: pd.DataFrame, field1: str, field2: str):
    columns = data[field1].unique().astype(str).tolist() + ["ALL"]
    rows = data[field2].unique().astype(str).tolist() + ["ALL"]
    real_linkage = pd.crosstab(data[field1], data[field2], margins=True).to_numpy()
    expected_linkage = np.round(contingency.expected_freq(real_linkage), 2)

    if real_linkage.shape[0] < real_linkage.shape[1]:
        real_linkage = np.transpose(real_linkage)
        expected_linkage = np.transpose(expected_linkage)
        real_linkage = add_labels(real_linkage, columns, [""] + rows)
        expected_linkage = add_labels(expected_linkage, columns, [""] + rows)
    else:
        real_linkage = add_labels(real_linkage, rows, [""] + columns)
        expected_linkage = add_labels(expected_linkage, rows, [""] + columns)

    return {
        "real_linkage": real_linkage,
        "expected_linkage": expected_linkage
        }


def chi2_correction(cross_table: pd.DataFrame):
    cross_table = cross_table.to_numpy()

    for ct_row in cross_table:
        for cell in ct_row:
            if float(cell) < 10:
                return False

    return True


def do_contingency_test(cross_table: pd.DataFrame) -> dict:
    test = chi2_contingency(cross_table, correction=chi2_correction(cross_table))

    return {
        "f": test[0],
        "p": test[1]
        }


def check_normality(column: pd.Series):
    '''
    Return Shapiro's test results on a sample.
    If the initial p-value is less than 0.05, do a log transformation and repeat.
    If p-value is less than 0.05, reduce sample size to a 100 and repeat.
    If p-value is less than 0.05, the normality test fails.
    '''
    print(type(column.values))
    shapiro_p = shapiro(column.values)[1]

    if shapiro_p > 0.05:
        return shapiro_p

    column_log = column.apply(lambda x: np.log(x + 1))
    shapiro_log_p = shapiro(column_log.values)[1]

    if shapiro_log_p > 0.05:
        return (shapiro_p, shapiro_log_p)

    column_cut = column.sample(100, ignore_index=True)
    shapiro_cut_p = shapiro(column_cut.values)[1]

    if shapiro_cut_p > 0.05:
        return (shapiro_p, shapiro_log_p, shapiro_cut_p, True)

    return (shapiro_p, shapiro_log_p, shapiro_cut_p, False)


def perform_anova(data: pd.DataFrame, field1: str, field2: str) -> dict:
    norm = check_normality(data[field1])
    if len(norm) == 2:
        data[field1] = data[field1].apply(lambda x: np.log(x + 1))
    elif len(norm) == 4:
        data = data.sample(100, ignore_index=True)

    groups = list()
    for unique in data[field2].unique():
        groups.append(data[data[field2] == unique][field1])

    f_anova = f_oneway(*groups)[1]
    levene_heteroskedasticity = levene(*groups)[1]

    model = ols(f"{field1} ~ {field2}", data).fit()
    durbin_watson_test = durbin_watson(model.resid)

    str_table_ = str(model.summary().tables[2])
    str_table_ = str_table_.replace("=", "", -1)
    str_table_ = str_table_.replace(": ", ":", -1)
    str_table_ = str_table_.replace("Cond. No. ", "Cond.No.:")
    str_table_ = str_table_.replace(" (JB)", "")
    list_table_ = str_table_.split()

    return {
        "anova_f": f_anova,
        "heteroskedasticity": levene_heteroskedasticity,
        "durbin_watson": durbin_watson_test,
        "additional_summary": [
            list_table_[:4],
            list_table_[4:8],
            list_table_[8:12],
            list_table_[12:]
            ],
    }
