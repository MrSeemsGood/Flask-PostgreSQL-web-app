from flask import Flask, redirect, url_for, request, render_template
from mystat import *
import os

class MyApp(Flask):
   def __init__(self, name, data):
      super().__init__(name)
      self.data = data
      self.tables = ()
      self.method = ""
      self.corr_test = None
      self.normality = False
      self.anova = None

app = MyApp(__name__, data=load_data())

@app.route('/')
def redirectToLogin():
   return redirect(url_for('login'))

# POST and GET are HTTP mehods.
# GET - Sends data in unencrypted form to the server. Most common method.
# POST - Used to send HTML form data to server. Data received by POST method is not cached by server.
@app.route('/login')
def login():
   return render_template('login.html', table=app.data.head().to_dict(orient='list'))

@app.route('/insert')
def insertValues():
   return render_template('insert.html')

@app.route('/test')
def chooseTests():
   return render_template('test.html', columnsList=app.data.columns)

@app.route('/insertresult', methods = ['POST', 'GET'])
def insertResult():
   if request.method == 'POST':
      result = dict(request.form)
   else:
      result = dict(request.args)

   app.data = pd.concat((app.data, pd.DataFrame(result, index=[0])))
   result_db = ','.join(["'" + str(item) + "'" for item in tuple(result.values())])
   connect_execute_db('INSERT INTO flaskdb VALUES (' + result_db + ')')
   return render_template('insertresult.html', result=result)

@app.route('/testresult', methods=['POST'])
def testResult():
   fields = list(dict(request.form).values())

   # таблицы сопряженности и ожидаемых значений
   try:
      app.tables = create_tables(app.data, fields[0], fields[1])
      app.method = choose_method(app.data, fields[0], fields[1])
      app.corr_test = perform_test(
         pd.crosstab(app.data[fields[0]], app.data[fields[1]], margins=True),
         app.method
      )
   except KeyError:
      return render_template('errorpage.html')
   
   return render_template(
         'testresult.html',
         fields=fields,
         linkage = app.tables['linkage'],
         expected=app.tables['expected'],
         method=app.method,
         F=app.corr_test['F'],
         p=app.corr_test['p'],
         result=app.corr_test['result']
      )

@app.route('/anovatestresult', methods=['POST'])
def anovaTestResult():
   fields = list(dict(request.form).values())

   try:
      app.normality = check_normality(app.data[fields[0]])
      app.anova = do_anova(app.data, fields[0], fields[1])
   except:
      return render_template('errorpage.html')
      
   return render_template(
      'anovatestresult.html',
      fields=fields,
      normality = app.normality,
      anova = app.anova
   )

if __name__ == '__main__':
   app.run(debug=True, port=2000)
