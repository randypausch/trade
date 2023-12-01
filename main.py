from flask import Flask, render_template, request, redirect, url_for, session, make_response, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, current_user, login_required
import pandas as pd
import plotly.express as px
import plotly.io as pio
import csv 
import io
from flask import Flask, render_template, request, session, make_response
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import plotly.io as pio
import csv 
import io
import os
import plotly.graph_objects as go
from flask import render_template, redirect, url_for, request, flash
from werkzeug.security import check_password_hash
from flask_login import login_user, logout_user, current_user, login_required
import random
from datetime import datetime, timedelta
import time



app = Flask(__name__)
app.secret_key = "your_secret_key_1"  # Change this to your secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

stocks_owned_1 = 100
stocks_owned_2 = 100
stocks_owned_3 = 100
stocks_owned_4 = 100
INITIAL_WALLET_AMOUNT = 99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
count = 0
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/", methods=["GET", "POST"])
def signup():
    # if current_user.is_authenticated:
    #     return redirect(url_for('intro')) 
    form = SignupForm()

    if form.validate_on_submit():
        # Check if the username already exists
        existing_user = User.query.filter_by(username=form.username.data).first()
        
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
        elif form.password.data != form.confirm_password.data:
            flash('Passwords do not match. Please try again.', 'danger')
        else:
            # Username is unique, proceed with registration
            hashed_password = generate_password_hash(form.password.data)
            user = User(username=form.username.data, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            flash('Account created successfully. You can now log in.', 'success')
            return redirect(url_for('login'))

    return render_template('signup.html', form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    # if current_user.is_authenticated:
    #     return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            df_random_generator()
            generate_data()
            generate_data1()
            return redirect(next_page) if next_page else redirect(url_for('form1'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    print("Before session clear:", session)
    session.clear()
    session.modified = True
    print("After session clear:", session)
    logout_user()
    print("After Logout:", session)
    return redirect(url_for('login'))


import numpy as np

# np.random.seed(42)

def df_random_generator():
    global df1, df2, df3, df4
    # List of file names
    file_names = ["stable_winning.csv", "stable_losing.csv", "volatile_winning.csv", "volatile_losing.csv"]

    # Randomly choose one unique file for each DataFrame
    selected_files = random.sample(file_names, k=4)

    df1 = pd.read_csv(selected_files[0]).round(2)
    df2 = pd.read_csv(selected_files[1]).round(2)
    df3 = pd.read_csv(selected_files[2]).round(2)
    df4 = pd.read_csv(selected_files[3]).round(2)
    # Extract information from file names and create variables
    volatile = [True if 'volatile' in file_name else False for file_name in selected_files]
    trend = ['up' if 'winning' in file_name else 'down' for file_name in selected_files]

    # Assign variables to DataFrames
    df1['volatile'] = volatile[0]
    df1['trend'] = trend[0]

    df2['volatile'] = volatile[1]
    df2['trend'] = trend[1]

    df3['volatile'] = volatile[2]
    df3['trend'] = trend[2]

    df4['volatile'] = volatile[3]
    df4['trend'] = trend[3]

def generate_data():
    global plotly_html, figure1, figure2, first_time
    # Map period to month names
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    # Initialize wallet amount to 1000 and store it in a session variable
    

    # Create line charts
    figure1 = px.scatter(df1.iloc[:0], x="Period", y="Base Price")
    figure1.data[0].update(mode='lines')
    figure2 = px.scatter(df2.iloc[:0], x="Period", y="Base Price")
    figure2.data[0].update(mode='lines')

    # Create a 1x2 subplot
    this_figure = sp.make_subplots(rows=1, cols=2, subplot_titles=["Security A", "Security B"])

    # Add traces to the subplo
    for trace in figure1.data:
        this_figure.add_trace(trace, row=1, col=1)

    for trace in figure2.data:
        this_figure.add_trace(trace, row=1, col=2)

    this_figure.update_layout(
        xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df1["Period"].min(), df1["Period"].max()], title="Period"),
        xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df2["Period"].min(), df2["Period"].max()], title="Period"),
        yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df1["Base Price"].min()-2, df1["Base Price"].max()+2]),
        yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df2["Base Price"].min()-2, df2["Base Price"].max()+2]),
        showlegend=False  # Hide legend
    )

    frames = [go.Frame(data=[go.Scatter(x = df1.iloc[:i]["Period"], y=df1.iloc[:i]["Base Price"]),
                            go.Scatter(x = df2.iloc[:i]["Period"], y=df2.iloc[:i]["Base Price"])],
                    name=f'Frame {i + 1}') for i in range(1, 21)]
    this_figure.frames = frames



    # Save the plot as HTML
    plotly_html = pio.to_html(this_figure, full_html=False, config = {'displayModeBar': False})


def generate_data1():
    global df3, df4, plotly_html_1, figure3, figure4

    # Map period to month names
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    # Initialize wallet amount to 1000 and store it in a session variable
    

    # Create line charts
    figure3 = px.scatter(df3.iloc[:0], x="Period", y="Base Price")
    figure3.data[0].update(mode='lines')
    figure4 = px.scatter(df4.iloc[:0], x="Period", y="Base Price")
    figure4.data[0].update(mode='lines')

    # Create a 1x2 subplot
    this_figure1 = sp.make_subplots(rows=1, cols=2, subplot_titles=["Security C", "Security D"])

    # Add traces to the subplo
    for trace in figure3.data:
        this_figure1.add_trace(trace, row=1, col=1)

    for trace in figure4.data:
        this_figure1.add_trace(trace, row=1, col=2)

    this_figure1.update_layout(
        xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df3["Period"].min(), df3["Period"].max()], title="Period"),
        xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df4["Period"].min(), df4["Period"].max()], title="Period"),
        yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df3["Base Price"].min()-2, df3["Base Price"].max()+2]),
        yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df4["Base Price"].min()-2, df4["Base Price"].max()+2]),
        showlegend=False  # Hide legend
    )

    frames = [go.Frame(data=[go.Scatter(x = df3.iloc[:i]["Period"], y=df3.iloc[:i]["Base Price"]),
                            go.Scatter(x = df4.iloc[:i]["Period"], y=df4.iloc[:i]["Base Price"])],
                    name=f'Frame {i + 1}') for i in range(1, 21)]
    this_figure1.frames = frames



    # Save the plot as HTML
    plotly_html_1 = pio.to_html(this_figure1, full_html=False, config = {'displayModeBar': False})
    
@app.route("/final")
def final():
    # Keys and results
    keys = ["trading_results_1", "trading_results_2", "trading_results_3", "trading_results_4"]
    dfs = [pd.DataFrame(session.get(key, [])) for key in keys]

    # Determine volatility of each DataFrame
    is_volatile = [df1['volatile'].any(), df2['volatile'].any(), df3['volatile'].any(), df4['volatile'].any()]

    # Separate DataFrames based on volatility
    market1_dfs = [df for df, volatile in zip(dfs, is_volatile) if volatile]  # Volatile
    market2_dfs = [df for df, volatile in zip(dfs, is_volatile) if not volatile]  # Stable

    # Compute COI and net profits for each market
    market1_coi = sum(100 * 100 + (df["Buy"] * df["Current_Price"]).sum() for df in market1_dfs).round(2)
    market1_net_profits = sum(df['Profit or Loss'].sum() for df in market1_dfs).round(2)

    market2_coi = sum(100 * 100 + (df["Buy"] * df["Current_Price"]).sum() for df in market2_dfs).round(2)
    market2_net_profits = sum(df['Profit or Loss'].sum() for df in market2_dfs).round(2)

    # Compute ROIs for each market
    market_1_roi = (market1_net_profits / market1_coi) * 100 if market1_coi != 0 else 0
    market_2_roi = (market2_net_profits / market2_coi) * 100 if market2_coi != 0 else 0

    # Compute combined ROI
    combined_roi = ((market1_net_profits + market2_net_profits) / (market1_coi + market2_coi)) * 100 if (market1_coi + market2_coi) != 0 else 0

    return render_template("final_scores.html", market_1_roi=market_1_roi.round(2), market_2_roi=market_2_roi.round(2), combined_roi=combined_roi.round(2))


@app.route("/intro")
def intro():
    return render_template("intro.html")

@app.route("/form1")
def form1():
    print("After Form:", session)
    return render_template("form.html")

@app.route("/form2")
def form2():
    return render_template("form2.html")

@app.route("/middle")
def middle():
    return render_template("gamemiddle.html")

# Flask route to render the HTML
@login_required
@app.route("/trade", methods=["GET", "POST"])
def index():
    # print("After index:", session)
    if request.method == "POST":
        # start_time = session.get("start_time", time.time())
        if request.form.get("action"):
            return perform_action()

    # Retrieve wallet amount from session
    wallet_amount_1 = session.get("wallet_amount_1", INITIAL_WALLET_AMOUNT)
    wallet_amount_2 = session.get("wallet_amount_2", INITIAL_WALLET_AMOUNT)

    stocks_owned_1 = session.get("stocks_owned_1", 100)
    stocks_owned_2 = session.get("stocks_owned_2", 100)
    session["trading_results_1"] = []
    session["trading_results_2"] = []
    
    


    # Calculate the number of stocks owned (assuming a fixed stock price)
    
    # Render the template with the wallet amount and other details
    return render_template(
        "index.html",
        plot=plotly_html,
        wallet_amount_1=wallet_amount_1,
        current_price_1=df1.iloc[19]["Base Price"],
        lowest_price_1=df1.iloc[:19]["Base Price"].min(),
        highest_price_1=df1.iloc[:19]["Base Price"].max(),
        stocks_owned_1=stocks_owned_1,
        wallet_amount_2=wallet_amount_2,
        current_price_2=df2.iloc[19]["Base Price"],
        lowest_price_2=df2.iloc[:19]["Base Price"].min(),
        highest_price_2=df2.iloc[:19]["Base Price"].max(),
        stocks_owned_2=stocks_owned_2
    )


# Flask route to handle buying, selling, or holding stocks
@app.route("/perform_action/<int:form_id>", methods=["GET","POST"])
def perform_action(form_id):
    if request.method == "GET":
        
        return index()
    global figure1, figure2, stocks_owned_1, stocks_owned_2  # Declare figures as global variables
    # print(request.form)
    # Retrieve the selected action from the form data
    selected_action = request.form[f"action_{form_id}"]
    # print(selected_action)
    if selected_action == "buy":
        return buy_stocks(form_id)
    elif selected_action == "sell":
        return sell_stocks(form_id)
    elif selected_action == "hold":
        return hold_stocks(form_id)
    else:
        return "Invalid action"

@app.route("/download_csv/<int:form_id>")
def download_csv(form_id):
    if form_id == 1:
        current_df = df1
    else:
        current_df = df2
    key = f"trading_results_{form_id}"
    trading_results = session.get(key, [])
    # print("Length of trading_results:", len(trading_results2))
    if len(trading_results) == 12:
        if not os.path.exists("experiment_results"):
            os.makedirs("experiment_results")
        # Create CSV file in memory
        current_timestamp = datetime.now()
        one_minute_ago = current_timestamp - timedelta(minutes=1)
        files_in_last_minute = [
            filename for filename in os.listdir("experiment_results")
            if current_user.username in filename and f"scenario{form_id}" in filename
            and os.path.getmtime(os.path.join("experiment_results", filename)) > one_minute_ago.timestamp()
        ]
        if files_in_last_minute:
            return "Another file was created in the last minute. Please wait before downloading again."
        tmp_df = pd.DataFrame(trading_results)

        coi = 100 * 100 * (tmp_df["No of Securities"]*tmp_df["Current_Price"])
        net_profit = tmp_df["Profit or Loss"].sum()
        roi = (net_profit/coi)*100
        volatility = current_df["volatile"]
        trend = current_df["trend"]
        tmp_df["Total ROI"] = roi
        tmp_df["Volatility"] = volatility
        tmp_df["Trend"] = trend

        csv_data = io.StringIO()
        tmp_df.to_csv(csv_data, index=False)

        # Get the current timestamp
        current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Save CSV data to a temporary file with the current timestamp
        temp_filename = f"experiment_results/{current_user.username}scenario{form_id}_{current_timestamp}.csv"
        with open(temp_filename, "w") as temp_file:
            temp_file.write(csv_data.getvalue())
    


        # Send the file as a response and remove the temporary file afterward
        return send_file(temp_filename, as_attachment=True, download_name=f"{temp_filename}", mimetype="text/csv", conditional=True)
    else:
        return "No trading results to download."

def buy_stocks(form_id):
    global figure1, figure2, stocks_owned_1 ,stocks_owned_2

     # Declare figures as global variables

    # Retrieve current wallet amount from session

    wallet_amount_1 = session.get("wallet_amount_1", INITIAL_WALLET_AMOUNT)
    wallet_amount_2 = session.get("wallet_amount_2", INITIAL_WALLET_AMOUNT)
    if form_id == 1:
            # Retrieve the current stock price
        figure1.data[0].y = df1.iloc[:20+len(figure1.data[0].y)]
        current_stock_price_1 = df1.iloc[: len(figure1.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_1 = df1.iloc[len(figure1.data[0].y) - 20: len(figure1.data[0].y)]["Base Price"].min()
        highest_price_1 = df1.iloc[len(figure1.data[0].y) - 20: len(figure1.data[0].y)]["Base Price"].max()

        current_stock_price_2 = df2.iloc[:20+ len(figure2.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_2 = df2.iloc[len(figure2.data[0].y):20+ len(figure2.data[0].y)]["Base Price"].min()
        highest_price_2 = df2.iloc[len(figure2.data[0].y):20+ len(figure2.data[0].y)]["Base Price"].max()
        stocks_to_buy = int(request.form["stocks_to_buy_1"])
        # Calculate the cost of buying stocks
        cost = stocks_to_buy * current_stock_price_1
        # Check if the wallet has sufficient funds
        if wallet_amount_1 >= cost:
            # Update wallet amount after buying stocks
            wallet_amount_1 -= cost
            stocks_owned_1 += stocks_to_buy
            session["stocks_owned_1"] = stocks_owned_1
            # Save updated wallet amount in session
            session["wallet_amount_1"] = wallet_amount_1

            update_session_1("Buy", current_stock_price_1, stocks_to_buy, 0, 0)

            # Update the graphs with the next 20 entries
            # print(figure1.data[0].y)
            figure1 = px.scatter(
                df1.iloc[: len(figure1.data[0].y)],
                x="Period",
                y="Base Price",
                title="Share Prices over time",
            )
            figure1.data[0].update(mode='lines')

            figure2 = px.scatter(
                df2.iloc[:len(figure2.data[0].y)],
                x="Period",
                y="Base Price",
                title="Share Prices over time",
            )
            figure2.data[0].update(mode='lines')

            # Create a 1x2 subplot
            this_figure = sp.make_subplots(
                rows=1, cols=2, subplot_titles=["Security A", "Security B"]
            )

            # Add traces to the subplot
            for trace in figure1.data:
                this_figure.add_trace(trace, row=1, col=1)

            for trace in figure2.data:
                this_figure.add_trace(trace, row=1, col=2)

            # Update x-axis tick positions and labels to start of each month
            tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
            tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

            this_figure.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df1["Period"].min(), df1["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df2["Period"].min(), df2["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df1["Base Price"].min()-2, df1["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df2["Base Price"].min()-2, df2["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

            # this_figure.update_layout(title_text="Share Prices over time")
            frames = [go.Frame(data=[go.Scatter(x = df1.iloc[:len(figure1.data[0].y)]["Period"], y=df1.iloc[:len(figure1.data[0].y)]["Base Price"]), 
                                     go.Scatter(x = df2.iloc[:i]["Period"], y=df2.iloc[:i]["Base Price"])],
                   name=f'Frame {i + 1}') for i in range(1+len(figure2.data[0].y), 21+len(figure2.data[0].y))]
            this_figure.frames = frames
            
            

            # Save the plot as HTML
            plotly_html = pio.to_html(this_figure, full_html=False, config = {'displayModeBar': False})
            # 



            return render_template(
                "index.html",
                plot=plotly_html,
                wallet_amount_1=wallet_amount_1,
                current_price_1=current_stock_price_1,
                lowest_price_1=lowest_price_1,
                highest_price_1=highest_price_1,
                stocks_owned_1=stocks_owned_1,
                wallet_amount_2=wallet_amount_2,
                current_price_2=current_stock_price_2, 
                lowest_price_2=lowest_price_2,
                highest_price_2=highest_price_2,
                stocks_owned_2=stocks_owned_2
            )
        else:
            return "Insufficient funds to buy stocks!"
    else:
        # Retrieve the current stock price
        figure2.data[0].y = df2.iloc[:20+ len(figure2.data[0].y)]
        current_stock_price_1 = df1.iloc[:20+ len(figure1.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_1 = df1.iloc[len(figure1.data[0].y):20+ len(figure1.data[0].y)]["Base Price"].min()
        highest_price_1 = df1.iloc[len(figure1.data[0].y):20+ len(figure1.data[0].y)]["Base Price"].max()

        current_stock_price_2 = df2.iloc[:len(figure2.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_2 = df2.iloc[len(figure2.data[0].y) - 20: len(figure2.data[0].y)]["Base Price"].min()
        highest_price_2 = df2.iloc[len(figure2.data[0].y) - 20 : len(figure2.data[0].y)]["Base Price"].max()
        stocks_to_buy = int(request.form["stocks_to_buy_2"])
        cost = stocks_to_buy * current_stock_price_2
        if wallet_amount_2 >= cost:
            # Update wallet amount after buying stocks
            wallet_amount_2 -= cost
            stocks_owned_2 += stocks_to_buy
            session["stocks_owned_2"] = stocks_owned_2
            # Save updated wallet amount in session
            session["wallet_amount_2"] = wallet_amount_2

            update_session_2("Buy", current_stock_price_2, stocks_to_buy, 0, 0)

            # Update the graphs with the next 20 entries
            figure1 = px.line(
                df1.iloc[:len(figure1.data[0].y)],
                x="Period",
                y="Base Price",
                title="Share Prices over time",
            )
            figure2 = px.line(
                df2.iloc[: len(figure2.data[0].y)],
                x="Period",
                y="Base Price",
                title="Share Prices over time",
            )

            # Create a 1x2 subplot
            this_figure = sp.make_subplots(
                rows=1, cols=2, subplot_titles=["Security A", "Security B"]
            )

            # Add traces to the subplot
            for trace in figure1.data:
                this_figure.add_trace(trace, row=1, col=1)

            for trace in figure2.data:
                this_figure.add_trace(trace, row=1, col=2)

            # Update x-axis tick positions and labels to start of each month

# Update x-axis tick positions and labels to start of each month
            tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
            tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

            this_figure.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df1["Period"].min(), df1["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df2["Period"].min(), df2["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df1["Base Price"].min()-2, df1["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df2["Base Price"].min()-2, df2["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

            # this_figure.update_layout(title_text="Share Prices over time")
            frames = [go.Frame(data=[go.Scatter(x = df1.iloc[:i]["Period"], y=df1.iloc[:i]["Base Price"]), 
                                     go.Scatter(x = df2.iloc[:len(figure2.data[0].y)]["Period"], y=df2.iloc[:len(figure2.data[0].y)]["Base Price"])],
                   name=f'Frame {i + 1}') for i in range(1+len(figure1.data[0].y), 21+len(figure1.data[0].y))]
            this_figure.frames = frames
            # this_figure.update_layout(title_text="Share Prices over time")

            # Save the plot as HTML
            plotly_html = pio.to_html(this_figure, full_html=False, config = {'displayModeBar': False})

            return render_template(
                "index.html",
                plot=plotly_html,
                wallet_amount_2=wallet_amount_2,
                current_price_2=current_stock_price_2,
                lowest_price_2=lowest_price_2,
                highest_price_2=highest_price_2,
                stocks_owned_2=stocks_owned_2,
                wallet_amount_1=wallet_amount_1,
                current_price_1=current_stock_price_1,
                lowest_price_1=lowest_price_1,
                highest_price_1=highest_price_1,
                stocks_owned_1=stocks_owned_1
            )
        else:
            return "Insufficient funds to buy stocks!"
        

def sell_stocks(form_id):
    global figure1, figure2, stocks_owned_1, stocks_owned_2 # Declare figures as global variables

    # Retrieve current wallet amount from session
    wallet_amount_1 = session.get("wallet_amount_1", INITIAL_WALLET_AMOUNT)
    wallet_amount_2 = session.get("wallet_amount_2", INITIAL_WALLET_AMOUNT)

    # Retrieve the amount of stocks to sell from the form data
    if form_id == 1:
        figure1.data[0].y = df1.iloc[:20+len(figure1.data[0].y)]
        current_stock_price_1 = df1.iloc[: len(figure1.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_1 = df1.iloc[len(figure1.data[0].y) - 20: len(figure1.data[0].y)]["Base Price"].min()
        highest_price_1 = df1.iloc[len(figure1.data[0].y) - 20: len(figure1.data[0].y)]["Base Price"].max()

        current_stock_price_2 = df2.iloc[:20+ len(figure2.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_2 = df2.iloc[len(figure2.data[0].y):20+ len(figure2.data[0].y)]["Base Price"].min()
        highest_price_2 = df2.iloc[len(figure2.data[0].y):20+ len(figure2.data[0].y)]["Base Price"].max()
        stocks_to_sell = int(request.form["stocks_to_sell_1"])
        earnings = stocks_to_sell * current_stock_price_1
        wallet_amount_1 += earnings
        stocks_owned_1 -= stocks_to_sell
        session["wallet_amount_1"] = wallet_amount_1
        session["stocks_owned_1"] = stocks_owned_1
        update_session_1("Sell", current_stock_price_1, 0, stocks_to_sell, 0)
        
        figure1 = px.scatter(
            df1.iloc[: len(figure1.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure1.data[0].update(mode='lines')

        figure2 = px.scatter(
            df2.iloc[:len(figure2.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure2.data[0].update(mode='lines')

        # Create a 1x2 subplot
        this_figure = sp.make_subplots(
            rows=1, cols=2, subplot_titles=["Security A", "Security B"]
        )

        # Add traces to the subplot
        for trace in figure1.data:
            this_figure.add_trace(trace, row=1, col=1)

        for trace in figure2.data:
            this_figure.add_trace(trace, row=1, col=2)

        # Update x-axis tick positions and labels to start of each month
        tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
        tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

        this_figure.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df1["Period"].min(), df1["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df2["Period"].min(), df2["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df1["Base Price"].min()-2, df1["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df2["Base Price"].min()-2, df2["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

        # this_figure.update_layout(title_text="Share Prices over time")
        frames = [go.Frame(data=[go.Scatter(x = df1.iloc[:len(figure1.data[0].y)]["Period"], y=df1.iloc[:len(figure1.data[0].y)]["Base Price"]), 
                                    go.Scatter(x = df2.iloc[:i]["Period"], y=df2.iloc[:i]["Base Price"])],
                name=f'Frame {i + 1}') for i in range(1+len(figure2.data[0].y), 21+len(figure2.data[0].y))]
        this_figure.frames = frames
        # this_figure.update_layout(title_text="Share Prices over time")

        # Save the plot as HTML
        plotly_html = pio.to_html(this_figure, full_html=False, config = {'displayModeBar': False})

        return render_template(
            "index.html",
            plot=plotly_html,
            wallet_amount_1=wallet_amount_1,
            current_price_1=current_stock_price_1,
            lowest_price_1=lowest_price_1,
            highest_price_1=highest_price_1,  
            stocks_owned_1=stocks_owned_1,
            wallet_amount_2=wallet_amount_2,
            current_price_2=current_stock_price_2,
            lowest_price_2=lowest_price_2,
            highest_price_2=highest_price_2,
            stocks_owned_2=stocks_owned_2
        )
    else:
        figure2.data[0].y = df2.iloc[:20+ len(figure2.data[0].y)]
        current_stock_price_1 = df1.iloc[:20+ len(figure1.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_1 = df1.iloc[len(figure1.data[0].y):20+ len(figure1.data[0].y)]["Base Price"].min()
        highest_price_1 = df1.iloc[len(figure1.data[0].y):20+ len(figure1.data[0].y)]["Base Price"].max()

        current_stock_price_2 = df2.iloc[:len(figure2.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_2 = df2.iloc[len(figure2.data[0].y) - 20: len(figure2.data[0].y)]["Base Price"].min()
        highest_price_2 = df2.iloc[len(figure2.data[0].y) - 20 : len(figure2.data[0].y)]["Base Price"].max()
        stocks_to_sell = int(request.form["stocks_to_sell_2"])
        earnings = stocks_to_sell * current_stock_price_2
        wallet_amount_2 += earnings
        stocks_owned_2 -= stocks_to_sell
        session["wallet_amount_2"] = wallet_amount_2
        session["stocks_owned_2"] = stocks_owned_2
        update_session_2("Sell", current_stock_price_2, 0, stocks_to_sell, 0)
            # Update the graphs with the next 20 entries
        figure1 = px.line(
            df1.iloc[:len(figure1.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure2 = px.line(
            df2.iloc[: len(figure2.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )

        # Create a 1x2 subplot
        this_figure = sp.make_subplots(
            rows=1, cols=2, subplot_titles=["Security A", "Security B"]
        )

        # Add traces to the subplot
        for trace in figure1.data:
            this_figure.add_trace(trace, row=1, col=1)

        for trace in figure2.data:
            this_figure.add_trace(trace, row=1, col=2)

        # Update x-axis tick positions and labels to start of each month

# Update x-axis tick positions and labels to start of each month
        tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
        tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

        this_figure.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df1["Period"].min(), df1["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df2["Period"].min(), df2["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df1["Base Price"].min()-2, df1["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df2["Base Price"].min()-2, df2["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

        # this_figure.update_layout(title_text="Share Prices over time")
        frames = [go.Frame(data=[go.Scatter(x = df1.iloc[:i]["Period"], y=df1.iloc[:i]["Base Price"]), 
                                    go.Scatter(x = df2.iloc[:len(figure2.data[0].y)]["Period"], y=df2.iloc[:len(figure2.data[0].y)]["Base Price"])],
                name=f'Frame {i + 1}') for i in range(1+len(figure1.data[0].y), 21+len(figure1.data[0].y))]
        this_figure.frames = frames


        # Save the plot as HTML
        plotly_html = pio.to_html(this_figure, full_html=False, config = {'displayModeBar': False})

        return render_template(
            "index.html",
            plot=plotly_html,
            wallet_amount_2=wallet_amount_2,
            current_price_2=current_stock_price_2,
            lowest_price_2=lowest_price_2,
            highest_price_2=highest_price_2,
            stocks_owned_2=stocks_owned_2,
            wallet_amount_1=wallet_amount_1,
            current_price_1=current_stock_price_1,
            lowest_price_1=lowest_price_1,
            highest_price_1=highest_price_1,
            stocks_owned_1=stocks_owned_1
        )
    


def hold_stocks(form_id):
    global figure1, figure2, stocks_owned_1, stocks_owned_2  # Declare figures as global variables

    # Retrieve current wallet amount from session
    wallet_amount_1 = session.get("wallet_amount", INITIAL_WALLET_AMOUNT)
    wallet_amount_2 = session.get("wallet_amount", INITIAL_WALLET_AMOUNT)

    if form_id == 1:
        session["stocks_owned_1"] = stocks_owned_1
        figure1.data[0].y = df1.iloc[:20+len(figure1.data[0].y)]
        current_stock_price_1 = df1.iloc[: len(figure1.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_1 = df1.iloc[len(figure1.data[0].y) - 20: len(figure1.data[0].y)]["Base Price"].min()
        highest_price_1 = df1.iloc[len(figure1.data[0].y) - 20: len(figure1.data[0].y)]["Base Price"].max()

        current_stock_price_2 = df2.iloc[:20+ len(figure2.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_2 = df2.iloc[len(figure2.data[0].y):20+ len(figure2.data[0].y)]["Base Price"].min()
        highest_price_2 = df2.iloc[len(figure2.data[0].y):20+ len(figure2.data[0].y)]["Base Price"].max()
        update_session_1("Hold", current_stock_price_1, 0, 0, stocks_owned_1 if form_id == 1 else stocks_owned_2)
            # Update the graphs with the next 20 entries
        figure1 = px.scatter(
            df1.iloc[: len(figure1.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure1.data[0].update(mode='lines')

        figure2 = px.scatter(
            df2.iloc[:len(figure2.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure2.data[0].update(mode='lines')

        # Create a 1x2 subplot
        this_figure = sp.make_subplots(
            rows=1, cols=2, subplot_titles=["Security A", "Security B"]
        )

        # Add traces to the subplot
        for trace in figure1.data:
            this_figure.add_trace(trace, row=1, col=1)

        for trace in figure2.data:
            this_figure.add_trace(trace, row=1, col=2)

        # Update x-axis tick positions and labels to start of each month
        tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
        tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

        this_figure.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df1["Period"].min(), df1["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df2["Period"].min(), df2["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df1["Base Price"].min()-2, df1["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df2["Base Price"].min()-2, df2["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

        # this_figure.update_layout(title_text="Share Prices over time")
        frames = [go.Frame(data=[go.Scatter(x = df1.iloc[:len(figure1.data[0].y)]["Period"], y=df1.iloc[:len(figure1.data[0].y)]["Base Price"]), 
                                    go.Scatter(x = df2.iloc[:i]["Period"], y=df2.iloc[:i]["Base Price"])],
                name=f'Frame {i + 1}') for i in range(1+len(figure2.data[0].y), 21+len(figure2.data[0].y))]
        this_figure.frames = frames
        # this_figure.update_layout(title_text="Share Prices over time")

        # Save the plot as HTML
        plotly_html = pio.to_html(this_figure, full_html=False, config = {'displayModeBar': False})

        return render_template(
            "index.html",
            plot=plotly_html,
            wallet_amount_1=wallet_amount_1,
            current_price_1=current_stock_price_1,
            lowest_price_1=lowest_price_1,
            highest_price_1=highest_price_1,  
            stocks_owned_1=stocks_owned_1,
            wallet_amount_2=wallet_amount_2,
            current_price_2=current_stock_price_2,
            lowest_price_2=lowest_price_2,
            highest_price_2=highest_price_2,
            stocks_owned_2=stocks_owned_2
        )

    else:
        session["stocks_owned_2"] = stocks_owned_2
        figure2.data[0].y = df2.iloc[:20+ len(figure2.data[0].y)]
        current_stock_price_1 = df1.iloc[:20+ len(figure1.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_1 = df1.iloc[len(figure1.data[0].y):20+ len(figure1.data[0].y)]["Base Price"].min()
        highest_price_1 = df1.iloc[len(figure1.data[0].y):20+ len(figure1.data[0].y)]["Base Price"].max()

        current_stock_price_2 = df2.iloc[:len(figure2.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_2 = df2.iloc[len(figure2.data[0].y) - 20: len(figure2.data[0].y)]["Base Price"].min()
        highest_price_2 = df2.iloc[len(figure2.data[0].y) - 20 : len(figure2.data[0].y)]["Base Price"].max()
        update_session_2("Hold", current_stock_price_2, 0, 0, stocks_owned_1 if form_id == 1 else stocks_owned_2)
            # Update the graphs with the next 20 entries
        figure1 = px.line(
            df1.iloc[:len(figure1.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure2 = px.line(
            df2.iloc[: len(figure2.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )

        # Create a 1x2 subplot
        this_figure = sp.make_subplots(
            rows=1, cols=2, subplot_titles=["Security A", "Security B"]
        )

        # Add traces to the subplot
        for trace in figure1.data:
            this_figure.add_trace(trace, row=1, col=1)

        for trace in figure2.data:
            this_figure.add_trace(trace, row=1, col=2)

        # Update x-axis tick positions and labels to start of each month

# Update x-axis tick positions and labels to start of each month
        tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
        tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

        this_figure.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df1["Period"].min(), df1["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df2["Period"].min(), df2["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df1["Base Price"].min()-2, df1["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df2["Base Price"].min()-2, df2["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

        # this_figure.update_layout(title_text="Share Prices over time")
        frames = [go.Frame(data=[go.Scatter(x = df1.iloc[:i]["Period"], y=df1.iloc[:i]["Base Price"]), 
                                    go.Scatter(x = df2.iloc[:len(figure2.data[0].y)]["Period"], y=df2.iloc[:len(figure2.data[0].y)]["Base Price"])],
                name=f'Frame {i + 1}') for i in range(1+len(figure1.data[0].y), 21+len(figure1.data[0].y))]
        this_figure.frames = frames


        # this_figure.update_layout(title_text="Share Prices over time")

        # Save the plot as HTML
        plotly_html = pio.to_html(this_figure, full_html=False, config = {'displayModeBar': False})

        return render_template(
            "index.html",
            plot=plotly_html,
            wallet_amount_2=wallet_amount_2,
            current_price_2=current_stock_price_2,
            lowest_price_2=lowest_price_2,
            highest_price_2=highest_price_2,
            stocks_owned_2=stocks_owned_2,
            wallet_amount_1=wallet_amount_1,
            current_price_1=current_stock_price_1,
            lowest_price_1=lowest_price_1,
            highest_price_1=highest_price_1,
            stocks_owned_1=stocks_owned_1
        )

def update_session_1(action,current_price, stocks_bought, stocks_sold, stocks_held):

    data = {
        "Current_Price": current_price,
        "Buy": stocks_bought,
        "Hold": stocks_held,
        "Sell": stocks_sold,
        "No of Securities": session.get("stocks_owned_1",100),
        "Profit or Loss": round(stocks_sold * current_price - (stocks_sold*100), 2),
        "ROI": (round(stocks_bought * current_price - (stocks_bought*100), 2))/(session.get("stocks_owned_1",100)),
        "Time taken": time.time()
    }

    # Append results to the session list
    session["trading_results_1"].append(data)

def update_session_2(action,current_price, stocks_bought, stocks_sold, stocks_held):

    data = {
        "Current_Price": current_price,
        "Buy": stocks_bought,
        "Hold": stocks_held,
        "Sell": stocks_sold,
        "No of Securities": session.get("stocks_owned_2",100),
        "Profit or Loss": round(stocks_sold * current_price - (stocks_sold*100), 2),
        "ROI": (round(stocks_sold * current_price - (stocks_sold*100), 2))/(session.get("stocks_owned_2",100)),
        "Time taken": time.time()
    }

    # Append results to the session list
    session["trading_results_2"].append(data)


@login_required
@app.route("/trade1", methods=["GET", "POST"])
def index1():

    if request.method == "POST":
        if request.form.get("action"):
            return perform_action1() 

    # Retrieve wallet amount from session
    wallet_amount_3 = session.get("wallet_amount_3", INITIAL_WALLET_AMOUNT)
    wallet_amount_4 = session.get("wallet_amount_4", INITIAL_WALLET_AMOUNT)

    stocks_owned_3 = session.get("stocks_owned_3", 100)
    stocks_owned_4 = session.get("stocks_owned_4", 100)
    session["trading_results_3"] = []
    session["trading_results_4"] = []
    



    # Calculate the number of stocks owned (assuming a fixed stock price)
    
    # Render the template with the wallet amount and other details
    return render_template(
        "index1.html",
        plot=plotly_html_1,
        wallet_amount_1=wallet_amount_3,
        current_price_1=df3.iloc[19]["Base Price"],
        lowest_price_1=df3.iloc[:19]["Base Price"].min(),
        highest_price_1=df3.iloc[:19]["Base Price"].max(),
        stocks_owned_1=stocks_owned_3,
        wallet_amount_2=wallet_amount_4,
        current_price_2=df4.iloc[19]["Base Price"],
        lowest_price_2=df4.iloc[:19]["Base Price"].min(),
        highest_price_2=df4.iloc[:19]["Base Price"].max(),
        stocks_owned_2=stocks_owned_4
    )


# Flask route to handle buying, selling, or holding stocks
@app.route("/perform_action1/<int:form_id>", methods=["GET","POST"])
def perform_action1(form_id):
    if request.method == "GET":
        
        return index1()
    global figure3, figure4, stocks_owned_3, stocks_owned_4  # Declare figures as global variables
    # print(request.form)
    # Retrieve the selected action from the form data
    selected_action = request.form[f"action_{form_id}"]
    # print(selected_action)

    if selected_action == "buy":
        return buy_stocks1(form_id)
    elif selected_action == "sell":
        return sell_stocks1(form_id)
    elif selected_action == "hold":
        return hold_stocks1(form_id)
    else:
        return "Invalid action"

@app.route("/download_csv1/<int:form_id>")
def download_csv1(form_id):
    if form_id == 1:
        k = 3
    else:
        k = 4
    if form_id == 1:
        current_df = df3
    else:
        current_df = df4
    key = f"trading_results_{k}"
    trading_results = session.get(key, [])
    print(len(trading_results))
    print(form_id)
    # print("Length of trading_results:", len(trading_results2))
    if len(trading_results) == 12:
        if not os.path.exists("experiment_results"):
            os.makedirs("experiment_results")
        # Create CSV file in memory
        current_timestamp = datetime.now()
        one_minute_ago = current_timestamp - timedelta(minutes=1)
        files_in_last_minute = [
            filename for filename in os.listdir("experiment_results")
            if current_user.username in filename and f"scenario{k}" in filename
            and os.path.getmtime(os.path.join("experiment_results", filename)) > one_minute_ago.timestamp()
        ]
        if files_in_last_minute:
            return "Another file was created in the last minute. Please wait before downloading again."
        
        tmp_df = pd.DataFrame(trading_results)
        coi = 100 * 100 * (tmp_df["No of Securities"]*tmp_df["Current_Price"])
        net_profit = tmp_df["Profit or Loss"].sum()
        roi = (net_profit/coi)*100
        volatility = current_df["volatile"]
        trend = current_df["trend"]
        # Add the new columns to the tmp_df DataFrame
        tmp_df["Total ROI"] = roi
        tmp_df["Volatility"] = volatility
        tmp_df["Trend"] = trend

        csv_data = io.StringIO()
        tmp_df.to_csv(csv_data, index=False)

        # Get the current timestamp
        current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Save CSV data to a temporary file with the current timestamp
        temp_filename = f"experiment_results/{current_user.username}scenario{k}_{current_timestamp}.csv"
        with open(temp_filename, "w") as temp_file:
            temp_file.write(csv_data.getvalue())
    


        # Send the file as a response and remove the temporary file afterward
        return send_file(temp_filename, as_attachment=True, download_name=f"{temp_filename}", mimetype="text/csv", conditional=True)
    else:
        return "No trading results to download."

def buy_stocks1(form_id):
    global figure3, figure4, stocks_owned_3 ,stocks_owned_4

     # Declare figures as global variables

    # Retrieve current wallet amount from session

    wallet_amount_3 = session.get("wallet_amount_3", INITIAL_WALLET_AMOUNT)
    wallet_amount_4 = session.get("wallet_amount_4", INITIAL_WALLET_AMOUNT)
    if form_id == 1:
            # Retrieve the current stock price
        figure3.data[0].y = df3.iloc[:20+len(figure3.data[0].y)]
        current_stock_price_3 = df3.iloc[: len(figure3.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_3 = df3.iloc[len(figure3.data[0].y) - 20: len(figure3.data[0].y)]["Base Price"].min()
        highest_price_3 = df3.iloc[len(figure3.data[0].y) - 20: len(figure3.data[0].y)]["Base Price"].max()

        current_stock_price_4 = df4.iloc[:20+ len(figure4.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_4 = df4.iloc[len(figure4.data[0].y):20+ len(figure4.data[0].y)]["Base Price"].min()
        highest_price_4 = df4.iloc[len(figure4.data[0].y):20+ len(figure4.data[0].y)]["Base Price"].max()
        stocks_to_buy = int(request.form["stocks_to_buy_1"])
        # Calculate the cost of buying stocks
        cost = stocks_to_buy * current_stock_price_3
        # Check if the wallet has sufficient funds
        if wallet_amount_3 >= cost:
            # Update wallet amount after buying stocks
            wallet_amount_3 -= cost
            stocks_owned_3 += stocks_to_buy
            session["stocks_owned_3"] = stocks_owned_3
            # Save updated wallet amount in session
            session["wallet_amount_3"] = wallet_amount_3

            update_session_3("Buy", current_stock_price_3, stocks_to_buy, 0, 0)

            # Update the graphs with the next 20 entries
            # print(figure3.data[0].y)
            figure3 = px.scatter(
                df3.iloc[: len(figure3.data[0].y)],
                x="Period",
                y="Base Price",
                title="Share Prices over time",
            )
            figure3.data[0].update(mode='lines')

            figure4 = px.scatter(
                df4.iloc[:len(figure4.data[0].y)],
                x="Period",
                y="Base Price",
                title="Share Prices over time",
            )
            figure4.data[0].update(mode='lines')

            # Create a 1x2 subplot
            this_figure1 = sp.make_subplots(
                rows=1, cols=2, subplot_titles=["Security C", "Security D"]
            )

            # Add traces to the subplot
            for trace in figure3.data:
                this_figure1.add_trace(trace, row=1, col=1)

            for trace in figure4.data:
                this_figure1.add_trace(trace, row=1, col=2)

            # Update x-axis tick positions and labels to start of each month
            tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
            tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

            this_figure1.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df3["Period"].min(), df3["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df4["Period"].min(), df4["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df3["Base Price"].min()-2, df3["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df4["Base Price"].min()-2, df4["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

            # this_figure1.update_layout(title_text="Share Prices over time")
            frames = [go.Frame(data=[go.Scatter(x = df3.iloc[:len(figure3.data[0].y)]["Period"], y=df3.iloc[:len(figure3.data[0].y)]["Base Price"]), 
                                     go.Scatter(x = df4.iloc[:i]["Period"], y=df4.iloc[:i]["Base Price"])],
                   name=f'Frame {i + 1}') for i in range(1+len(figure4.data[0].y), 21+len(figure4.data[0].y))]
            this_figure1.frames = frames
            
            

            # Save the plot as HTML
            plotly_html_1 = pio.to_html(this_figure1, full_html=False, config = {'displayModeBar': False})
            # 

            return render_template(
                "index1.html",
                plot=plotly_html_1,
                wallet_amount_1=wallet_amount_3,
                current_price_1=current_stock_price_3,
                lowest_price_1=lowest_price_3,
                highest_price_1=highest_price_3,
                stocks_owned_1=stocks_owned_3,
                wallet_amount_2=wallet_amount_4,
                current_price_2=current_stock_price_4, 
                lowest_price_2=lowest_price_4,
                highest_price_2=highest_price_4,
                stocks_owned_2=stocks_owned_4
            )
        else:
            return "Insufficient funds to buy stocks!"
    else:
        # Retrieve the current stock price
        figure4.data[0].y = df4.iloc[:20+ len(figure4.data[0].y)]
        current_stock_price_3 = df3.iloc[:20+ len(figure3.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_3 = df3.iloc[len(figure3.data[0].y):20+ len(figure3.data[0].y)]["Base Price"].min()
        highest_price_3 = df3.iloc[len(figure3.data[0].y):20+ len(figure3.data[0].y)]["Base Price"].max()

        current_stock_price_4 = df4.iloc[:len(figure4.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_4 = df4.iloc[len(figure4.data[0].y) - 20: len(figure4.data[0].y)]["Base Price"].min()
        highest_price_4 = df4.iloc[len(figure4.data[0].y) - 20 : len(figure4.data[0].y)]["Base Price"].max()
        stocks_to_buy = int(request.form["stocks_to_buy_2"])
        cost = stocks_to_buy * current_stock_price_4
        if wallet_amount_4 >= cost:
            # Update wallet amount after buying stocks
            wallet_amount_4 -= cost
            stocks_owned_4 += stocks_to_buy
            session["stocks_owned_4"] = stocks_owned_4
            # Save updated wallet amount in session
            session["wallet_amount_4"] = wallet_amount_4

            update_session_4("Buy", current_stock_price_4, stocks_to_buy, 0, 0)

            # Update the graphs with the next 20 entries
            figure3 = px.line(
                df3.iloc[:len(figure3.data[0].y)],
                x="Period",
                y="Base Price",
                title="Share Prices over time",
            )
            figure4 = px.line(
                df4.iloc[: len(figure4.data[0].y)],
                x="Period",
                y="Base Price",
                title="Share Prices over time",
            )

            # Create a 1x2 subplot
            this_figure1 = sp.make_subplots(
                rows=1, cols=2, subplot_titles=["Security C", "Security D"]
            )

            # Add traces to the subplot
            for trace in figure3.data:
                this_figure1.add_trace(trace, row=1, col=1)

            for trace in figure4.data:
                this_figure1.add_trace(trace, row=1, col=2)

            # Update x-axis tick positions and labels to start of each month

# Update x-axis tick positions and labels to start of each month
            tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
            tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

            this_figure1.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df3["Period"].min(), df3["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df4["Period"].min(), df4["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df3["Base Price"].min()-2, df3["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df4["Base Price"].min()-2, df4["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

            # this_figure1.update_layout(title_text="Share Prices over time")
            frames = [go.Frame(data=[go.Scatter(x = df3.iloc[:i]["Period"], y=df3.iloc[:i]["Base Price"]), 
                                     go.Scatter(x = df4.iloc[:len(figure4.data[0].y)]["Period"], y=df4.iloc[:len(figure4.data[0].y)]["Base Price"])],
                   name=f'Frame {i + 1}') for i in range(1+len(figure3.data[0].y), 21+len(figure3.data[0].y))]
            this_figure1.frames = frames
            # this_figure1.update_layout(title_text="Share Prices over time")

            # Save the plot as HTML
            plotly_html_1 = pio.to_html(this_figure1, full_html=False, config = {'displayModeBar': False})

            return render_template(
                "index1.html",
                plot=plotly_html_1,
                wallet_amount_2=wallet_amount_4,
                current_price_2=current_stock_price_4,
                lowest_price_2=lowest_price_4,
                highest_price_2=highest_price_4,
                stocks_owned_2=stocks_owned_4,
                wallet_amount_1=wallet_amount_3,
                current_price_1=current_stock_price_3,
                lowest_price_1=lowest_price_3,
                highest_price_1=highest_price_3,
                stocks_owned_1=stocks_owned_3
            )
        else:
            return "Insufficient funds to buy stocks!"
        

def sell_stocks1(form_id):
    global figure3, figure4, stocks_owned_3, stocks_owned_4 # Declare figures as global variables

    # Retrieve current wallet amount from session
    wallet_amount_3 = session.get("wallet_amount_3", INITIAL_WALLET_AMOUNT)
    wallet_amount_4 = session.get("wallet_amount_4", INITIAL_WALLET_AMOUNT)

    # Retrieve the amount of stocks to sell from the form data
    if form_id == 1:
        figure3.data[0].y = df3.iloc[:20+len(figure3.data[0].y)]
        current_stock_price_3 = df3.iloc[: len(figure3.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_3 = df3.iloc[len(figure3.data[0].y) - 20: len(figure3.data[0].y)]["Base Price"].min()
        highest_price_3 = df3.iloc[len(figure3.data[0].y) - 20: len(figure3.data[0].y)]["Base Price"].max()

        current_stock_price_4 = df4.iloc[:20+ len(figure4.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_4 = df4.iloc[len(figure4.data[0].y):20+ len(figure4.data[0].y)]["Base Price"].min()
        highest_price_4 = df4.iloc[len(figure4.data[0].y):20+ len(figure4.data[0].y)]["Base Price"].max()
        stocks_to_sell = int(request.form["stocks_to_sell_1"])
        earnings = stocks_to_sell * current_stock_price_3
        wallet_amount_3 += earnings
        stocks_owned_3 -= stocks_to_sell
        session["wallet_amount_3"] = wallet_amount_3
        session["stocks_owned_3"] = stocks_owned_3
        update_session_3("Sell", current_stock_price_3, 0, stocks_to_sell, 0)
        
        figure3 = px.scatter(
            df3.iloc[: len(figure3.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure3.data[0].update(mode='lines')

        figure4 = px.scatter(
            df4.iloc[:len(figure4.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure4.data[0].update(mode='lines')

        # Create a 1x2 subplot
        this_figure1 = sp.make_subplots(
            rows=1, cols=2, subplot_titles=["Security C", "Security D"]
        )

        # Add traces to the subplot
        for trace in figure3.data:
            this_figure1.add_trace(trace, row=1, col=1)

        for trace in figure4.data:
            this_figure1.add_trace(trace, row=1, col=2)

        # Update x-axis tick positions and labels to start of each month
        tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
        tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

        this_figure1.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df3["Period"].min(), df3["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df4["Period"].min(), df4["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df3["Base Price"].min()-2, df3["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df4["Base Price"].min()-2, df4["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

        # this_figure1.update_layout(title_text="Share Prices over time")
        frames = [go.Frame(data=[go.Scatter(x = df3.iloc[:len(figure3.data[0].y)]["Period"], y=df3.iloc[:len(figure3.data[0].y)]["Base Price"]), 
                                    go.Scatter(x = df4.iloc[:i]["Period"], y=df4.iloc[:i]["Base Price"])],
                name=f'Frame {i + 1}') for i in range(1+len(figure4.data[0].y), 21+len(figure4.data[0].y))]
        this_figure1.frames = frames
        # this_figure1.update_layout(title_text="Share Prices over time")

        # Save the plot as HTML
        plotly_html_1 = pio.to_html(this_figure1, full_html=False, config = {'displayModeBar': False})

        return render_template(
            "index1.html",
            plot=plotly_html_1,
            wallet_amount_1=wallet_amount_3,
            current_price_1=current_stock_price_3,
            lowest_price_1=lowest_price_3,
            highest_price_1=highest_price_3,  
            stocks_owned_1=stocks_owned_3,
            wallet_amount_2=wallet_amount_4,
            current_price_2=current_stock_price_4,
            lowest_price_2=lowest_price_4,
            highest_price_2=highest_price_4,
            stocks_owned_2=stocks_owned_4
        )
    else:
        figure4.data[0].y = df4.iloc[:20+ len(figure4.data[0].y)]
        current_stock_price_3 = df3.iloc[:20+ len(figure3.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_3 = df3.iloc[len(figure3.data[0].y):20+ len(figure3.data[0].y)]["Base Price"].min()
        highest_price_3 = df3.iloc[len(figure3.data[0].y):20+ len(figure3.data[0].y)]["Base Price"].max()

        current_stock_price_4 = df4.iloc[:len(figure4.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_4 = df4.iloc[len(figure4.data[0].y) - 20: len(figure4.data[0].y)]["Base Price"].min()
        highest_price_4 = df4.iloc[len(figure4.data[0].y) - 20 : len(figure4.data[0].y)]["Base Price"].max()
        stocks_to_sell = int(request.form["stocks_to_sell_2"])
        earnings = stocks_to_sell * current_stock_price_4
        wallet_amount_4 += earnings
        stocks_owned_4 -= stocks_to_sell
        session["wallet_amount_4"] = wallet_amount_4
        session["stocks_owned_4"] = stocks_owned_4
        update_session_4("Sell", current_stock_price_4, 0, stocks_to_sell, 0)
            # Update the graphs with the next 20 entries
        figure3 = px.line(
            df3.iloc[:len(figure3.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure4 = px.line(
            df4.iloc[: len(figure4.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )

        # Create a 1x2 subplot
        this_figure1 = sp.make_subplots(
            rows=1, cols=2, subplot_titles=["Security C", "Security D"]
        )

        # Add traces to the subplot
        for trace in figure3.data:
            this_figure1.add_trace(trace, row=1, col=1)

        for trace in figure4.data:
            this_figure1.add_trace(trace, row=1, col=2)

        # Update x-axis tick positions and labels to start of each month

# Update x-axis tick positions and labels to start of each month
        tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
        tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

        this_figure1.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df3["Period"].min(), df3["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df4["Period"].min(), df4["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df3["Base Price"].min()-2, df3["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df4["Base Price"].min()-2, df4["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

        # this_figure1.update_layout(title_text="Share Prices over time")
        frames = [go.Frame(data=[go.Scatter(x = df3.iloc[:i]["Period"], y=df3.iloc[:i]["Base Price"]), 
                                    go.Scatter(x = df4.iloc[:len(figure4.data[0].y)]["Period"], y=df4.iloc[:len(figure4.data[0].y)]["Base Price"])],
                name=f'Frame {i + 1}') for i in range(1+len(figure3.data[0].y), 21+len(figure3.data[0].y))]
        this_figure1.frames = frames


        # Save the plot as HTML
        plotly_html_1 = pio.to_html(this_figure1, full_html=False, config = {'displayModeBar': False})

        return render_template(
            "index1.html",
            plot=plotly_html_1,
            wallet_amount_2=wallet_amount_4,
            current_price_2=current_stock_price_4,
            lowest_price_2=lowest_price_4,
            highest_price_2=highest_price_4,
            stocks_owned_2=stocks_owned_4,
            wallet_amount_1=wallet_amount_3,
            current_price_1=current_stock_price_3,
            lowest_price_1=lowest_price_3,
            highest_price_1=highest_price_3,
            stocks_owned_1=stocks_owned_3
        )
    


def hold_stocks1(form_id):
    global figure3, figure4, stocks_owned_3, stocks_owned_4  # Declare figures as global variables

    # Retrieve current wallet amount from session
    wallet_amount_3 = session.get("wallet_amount", INITIAL_WALLET_AMOUNT)
    wallet_amount_4 = session.get("wallet_amount", INITIAL_WALLET_AMOUNT)

    if form_id == 1:
        session["stocks_owned_3"] = stocks_owned_3
        figure3.data[0].y = df3.iloc[:20+len(figure3.data[0].y)]
        current_stock_price_3 = df3.iloc[: len(figure3.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_3 = df3.iloc[len(figure3.data[0].y) - 20: len(figure3.data[0].y)]["Base Price"].min()
        highest_price_3 = df3.iloc[len(figure3.data[0].y) - 20: len(figure3.data[0].y)]["Base Price"].max()

        current_stock_price_4 = df4.iloc[:20+ len(figure4.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_4 = df4.iloc[len(figure4.data[0].y):20+ len(figure4.data[0].y)]["Base Price"].min()
        highest_price_4 = df4.iloc[len(figure4.data[0].y):20+ len(figure4.data[0].y)]["Base Price"].max()
        update_session_3("Hold", current_stock_price_3, 0, 0, stocks_owned_3 if form_id == 1 else stocks_owned_4)
            # Update the graphs with the next 20 entries
        figure3 = px.scatter(
            df3.iloc[: len(figure3.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure3.data[0].update(mode='lines')

        figure4 = px.scatter(
            df4.iloc[:len(figure4.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure4.data[0].update(mode='lines')

        # Create a 1x2 subplot
        this_figure1 = sp.make_subplots(
            rows=1, cols=2, subplot_titles=["Security C", "Security D"]
        )

        # Add traces to the subplot
        for trace in figure3.data:
            this_figure1.add_trace(trace, row=1, col=1)

        for trace in figure4.data:
            this_figure1.add_trace(trace, row=1, col=2)

        # Update x-axis tick positions and labels to start of each month
        tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
        tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

        this_figure1.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df3["Period"].min(), df3["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df4["Period"].min(), df4["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df3["Base Price"].min()-2, df3["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df4["Base Price"].min()-2, df4["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

        # this_figure1.update_layout(title_text="Share Prices over time")
        frames = [go.Frame(data=[go.Scatter(x = df3.iloc[:len(figure3.data[0].y)]["Period"], y=df3.iloc[:len(figure3.data[0].y)]["Base Price"]), 
                                    go.Scatter(x = df4.iloc[:i]["Period"], y=df4.iloc[:i]["Base Price"])],
                name=f'Frame {i + 1}') for i in range(1+len(figure4.data[0].y), 21+len(figure4.data[0].y))]
        this_figure1.frames = frames
        # this_figure1.update_layout(title_text="Share Prices over time")

        # Save the plot as HTML
        plotly_html_1 = pio.to_html(this_figure1, full_html=False, config = {'displayModeBar': False})

        return render_template(
            "index1.html",
            plot=plotly_html_1,
            wallet_amount_1=wallet_amount_3,
            current_price_1=current_stock_price_3,
            lowest_price_1=lowest_price_3,
            highest_price_1=highest_price_3,  
            stocks_owned_1=stocks_owned_3,
            wallet_amount_2=wallet_amount_4,
            current_price_2=current_stock_price_4,
            lowest_price_2=lowest_price_4,
            highest_price_2=highest_price_4,
            stocks_owned_2=stocks_owned_4
        )

    else:
        session["stocks_owned_4"] = stocks_owned_4
        figure4.data[0].y = df4.iloc[:20+ len(figure4.data[0].y)]
        current_stock_price_3 = df3.iloc[:20+ len(figure3.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_3 = df3.iloc[len(figure3.data[0].y):20+ len(figure3.data[0].y)]["Base Price"].min()
        highest_price_3 = df3.iloc[len(figure3.data[0].y):20+ len(figure3.data[0].y)]["Base Price"].max()

        current_stock_price_4 = df4.iloc[:len(figure4.data[0].y)]["Base Price"].iloc[-1]
        lowest_price_4 = df4.iloc[len(figure4.data[0].y) - 20: len(figure4.data[0].y)]["Base Price"].min()
        highest_price_4 = df4.iloc[len(figure4.data[0].y) - 20 : len(figure4.data[0].y)]["Base Price"].max()
        update_session_4("Hold", current_stock_price_4, 0, 0, stocks_owned_3 if form_id == 1 else stocks_owned_4)
            # Update the graphs with the next 20 entries
        figure3 = px.line(
            df3.iloc[:len(figure3.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )
        figure4 = px.line(
            df4.iloc[: len(figure4.data[0].y)],
            x="Period",
            y="Base Price",
            title="Share Prices over time",
        )

        # Create a 1x2 subplot
        this_figure1 = sp.make_subplots(
            rows=1, cols=2, subplot_titles=["Security C", "Security D"]
        )

        # Add traces to the subplot
        for trace in figure3.data:
            this_figure1.add_trace(trace, row=1, col=1)

        for trace in figure4.data:
            this_figure1.add_trace(trace, row=1, col=2)

        # Update x-axis tick positions and labels to start of each month

# Update x-axis tick positions and labels to start of each month
        tick_positions = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
        tick_labels = months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

        this_figure1.update_layout(
    xaxis1=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df3["Period"].min(), df3["Period"].max()], title="Period"),
    xaxis2=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_labels,range=[df4["Period"].min(), df4["Period"].max()], title="Period"),
    yaxis1=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df3["Base Price"].min()-2, df3["Base Price"].max()+2]),
    yaxis2=dict(tickmode="array", tickvals=[100], ticktext=["100"],title="Prices",range=[df4["Base Price"].min()-2, df4["Base Price"].max()+2]),
    showlegend=False  # Hide legend
)

        # this_figure1.update_layout(title_text="Share Prices over time")
        frames = [go.Frame(data=[go.Scatter(x = df3.iloc[:i]["Period"], y=df3.iloc[:i]["Base Price"]), 
                                    go.Scatter(x = df4.iloc[:len(figure4.data[0].y)]["Period"], y=df4.iloc[:len(figure4.data[0].y)]["Base Price"])],
                name=f'Frame {i + 1}') for i in range(1+len(figure3.data[0].y), 21+len(figure3.data[0].y))]
        this_figure1.frames = frames


        # this_figure1.update_layout(title_text="Share Prices over time")

        # Save the plot as HTML
        plotly_html_1 = pio.to_html(this_figure1, full_html=False, config = {'displayModeBar': False})

        return render_template(
            "index1.html",
            plot=plotly_html_1,
            wallet_amount_2=wallet_amount_4,
            current_price_2=current_stock_price_4,
            lowest_price_2=lowest_price_4,
            highest_price_2=highest_price_4,
            stocks_owned_2=stocks_owned_4,
            wallet_amount_1=wallet_amount_3,
            current_price_1=current_stock_price_3,
            lowest_price_1=lowest_price_3,
            highest_price_1=highest_price_3,
            stocks_owned_1=stocks_owned_3
        )

def update_session_3(action,current_price, stocks_bought, stocks_sold, stocks_held):

    data = {
        "Current_Price": current_price,
        "Buy": stocks_bought,
        "Hold": stocks_held,
        "Sell": stocks_sold,
        "No of Securities": session.get("stocks_owned_3",100),
        "Profit or Loss": round(stocks_sold * current_price - (stocks_sold*100), 2),
        "ROI": (round(stocks_bought * current_price - (stocks_bought*100), 2))/(session.get("stocks_owned_3",100)),
        "Time taken": time.time()
    }

    # Append results to the session list
    session["trading_results_3"].append(data)
    

def update_session_4(action,current_price, stocks_bought, stocks_sold, stocks_held):

    data = {
        "Current_Price": current_price,
        "Buy": stocks_bought,
        "Hold": stocks_held,
        "Sell": stocks_sold,
        "No of Securities": session.get("stocks_owned_4",100),
        "Profit or Loss": round(stocks_sold * current_price - (stocks_sold*100), 2),
        "ROI": (round(stocks_sold * current_price - (stocks_sold*100), 2))/(session.get("stocks_owned_4",100)),
        "Time taken": time.time()
    }

    # Append results to the session list
    session["trading_results_4"].append(data)





if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)



