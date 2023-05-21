import pandas as pd
from flask import (
    Flask,
    redirect,
    url_for,
    request,
    render_template,
    session
)
from stat_work import (
    connect_execute_db,
    load_data,
    create_tables,
    do_contingency_test,
    check_normality,
    perform_anova
)


app = Flask(__name__)


@app.route("/")
def redirect_to_login():
    return redirect(url_for("login"))


@app.route("/login")
def login():
    session.clear()
    session['data_head'] = load_data().head().to_dict(orient='list')
    return render_template(
        "login.html",
        display_table=session['data_head']
        )


@app.route("/insert")
def insert_values():
    return render_template("insert.html")


@app.route("/test")
def select_tests():
    try:
        return render_template(
            "test.html",
            columns_list=pd.DataFrame(session['data_head']).columns
            )
    except KeyError:
        return redirect(url_for("login"))


@app.route("/insertresult", methods=["POST", "GET"])
def insert_result():
    if request.method == "POST":
        result = dict(request.form)
    else:
        result = dict(request.args)

    connect_execute_db("""
        INSERT INTO flaskdb VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        args=tuple(result.values())
        )

    return render_template(
        "insertresult.html",
        )


@app.route("/testresult", methods=["POST"])
def test_result():
    fields = list(dict(request.form).values())
    data = load_data()

    try:
        contingency = create_tables(data, fields[0], fields[1])
        contingency_test = do_contingency_test(
            pd.crosstab(
                data[fields[0]], data[fields[1]], margins=True
                )
        )
    except KeyError:
        return render_template("errorpage.html")

    return render_template(
        "testresult.html",
        fields=fields,
        linkage=contingency["real_linkage"],
        expected=contingency["expected_linkage"],
        F=contingency_test["f"],
        p=contingency_test["p"]
    )


@app.route("/anovatestresult", methods=["POST"])
def anova_test_result():
    fields = list(dict(request.form).values())
    data = load_data()

    try:
        normality = check_normality(data[fields[0]])
        anova = perform_anova(data, fields[0], fields[1])
    except KeyError:
        return render_template("errorpage.html")

    return render_template(
        "anovatestresult.html",
        fields=fields,
        normality=normality,
        anova=anova
    )


if __name__ == "__main__":
    app.config['SECRET_KEY'] = 'ThisIsSecretSecretSecret'
    app.run(port=2000)
