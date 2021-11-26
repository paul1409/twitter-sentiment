from flask import Flask, render_template, request, redirect, url_for
import tweepy
from keys import consumer_key, consumer_secret, access_token, access_token_secret
from model import setup, apply_prediction

# print(consumer_key, consumer_secret, access_token, access_token_secret)
# authorization


try:
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    me = api.me()
    name = me.screen_name
    tweets = api.user_timeline(name)
    setup()
except:
    me = None
    name = None
    tweets = None
    print('test')

# tweets = [
#     {
#         "id": 1,
#         "text": 'somestuff'
#     },
#     {
#         "id": 2,
#         "text": 'someotherstuff'
#     }
# ]


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        content = request.form['content']
        new_status = api.update_status(content)
        print(content)

    try:
        tweets = api.user_timeline(name)
    except:
        tweets = None
    return render_template("index.html", name=name, tweets=tweets)


@app.route("/tweet:<id>", methods=["GET", "POST"])
def get_single_tweet(id):

    sentiment = None
    try:
        tweet = [api.get_status(id)]
        for x in tweet:
            sentiment = apply_prediction(x.text)
    except:
        tweet = None
    return render_template("index.html", tweets=tweet, sentiment=sentiment)


@app.route("/status", methods=["GET", "POST"])
def post_status():
    return render_template("post_status.html", name=name)


@app.route("/remove_status:<id>", methods=["GET", "POST"])
def get_status(id):
    print(id)
    try:
        api.destroy_status(id)
    except:
        print('error')
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
