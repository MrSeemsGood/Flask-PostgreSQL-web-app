from flask import Flask, redirect, url_for, request, render_template
from mystat import *

class MyApp(Flask):
   def __init__(self, name, data):
      super().__init__(name)
      # загрузка данных
      self.data = data
      self.tables = ()
      self.method = ""
      self.corrTest = None
      self.normality = False
      self.anova = None

app = MyApp(__name__, data=load_data())

# POST and GET are HTTP mehods.
# GET - Sends data in unencrypted form to the server. Most common method.
# POST - Used to send HTML form data to server. Data received by POST method is not cached by server.
@app.route('/login')
def login():
   return render_template('login.html')

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
   return render_template('insertresult.html', result=result, new_len=app.data.shape[0])

@app.route('/testresult', methods=['POST'])
def testResult():
   fields = list(dict(request.form).values())

   # таблицы сопряженности и ожидаемых значений
   try:
      app.tables = create_tables(app.data, fields[0], fields[1])
      app.method = choose_method(app.data, fields[0], fields[1])
      app.corrTest = perform_test(
         pd.crosstab(app.data[fields[0]], app.data[fields[1]], margins=True),
         app.method
      )
   except KeyError:
      return render_template('errorpage.html')
   
   return render_template(
         'testresult.html', 
         field1=fields[0], 
         field2=fields[1], 
         link_t=app.tables[0], 
         exp=app.tables[1],
         method=app.method,
         fstat = app.corrTest[0],
         p = app.corrTest[1],
         itr = app.corrTest[2]
      )

@app.route('/anovatestresult', methods=['POST'])
def anovaTestResult():
   fields = list(dict(request.form).values())

   try:
      app.normality = check_normality(app.data[fields[0]])
      app.anova = anova(app.data, fields[0], fields[1])
   except:
      return render_template('errorpage.html')
      
   return render_template(
      'anovatestresult.html',
      field1=fields[0], 
      field2=fields[1],
      normality = app.normality,
      anova = app.anova
   )

if __name__ == '__main__':
   print('Запуск сервера...')
   print('тест коммит')
   app.run(debug=True)