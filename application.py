from math import e
from flask import Flask, render_template, request, redirect, url_for
import tweepy
# from keys import consumer_key, consumer_secret, access_token, access_token_secret
consumer_key = "iPNBiD61yXuISv8y9XWr2yzoC"
consumer_secret = "LACyQ2yJIWNtMAO8UlBOE0LLZSDXJOVaoswPnwWXd6w4jEm1ru"
access_token = "1311486910322286592-FOruybm5pHe0sePabEcwxu5EnPLu7D"
access_token_secret = "u8QRfPwwn1furI7RMqnItJ6Q5l07vjMdWs5tQ8gYpb68U"
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
    tweet_data = []
    try:
        tweets = api.user_timeline(name)
        
    except:
        tweets = None
    sentiments = []
    try:
        for tweet in tweets:
            sentiment = apply_prediction(tweet.text)
            sentiments.append(sentiment)
            tweet_data.append({
                "text": tweet.text,
                "id": tweet.id,
                "sentiment": sentiment
            })
        #print(tweet_data)
        tweets = tweet_data
        #print(tweets)
    except:
        print("something with apply prediction fxn")

    return render_template("index.html", name=name, tweets=tweets)


@app.route("/tweet:<id>", methods=["GET", "POST"])
def get_single_tweet(id):

    sentiment = None
    try:
        #print(id)
        tweet = [api.get_status(id)]
        #print(tweet)
        tweet = [{
            "text": tweet[0].text,
            "id": id,
            "sentiment": apply_prediction(tweet[0].text)
        }]
        print(tweet)

    except:
        tweet = None
        print('error getting tweet!!!!')

    # sentiment = apply_prediction(tweet[0].text)
    return render_template("index.html", tweets=tweet)


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
    app.run(debug=False)
