
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from bs4 import BeautifulSoup
import requests



# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27dsssf27567d441f2b6176a'


@app.route('/')
@app.route('/index')
def index():
    return render_template('yelpSearch.html')


# @app.route("/<string:q>/")
# def hello(q):
#     return render_template('yelpS.html', name=q)

class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])


@app.route('/Results', methods=['GET', 'POST'])
def result(result=None):
    if request.args.get('mail', None):
        result = request.args['mail']
        print(result)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        div = soup.find_all('div', {"class": "review-content"})

        reviews = []
        for d in div:
            reviews.append(d.find('p').text)

        # print(soup)

    return render_template('yelpResult.html', result=result)

if __name__ == "__main__":
    app.run()

