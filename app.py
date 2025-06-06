from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///receipts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# グローバル変数で簡易的に保持（本来はDBに保存すべき）
monthly_budget = 1000.0
savings_goal = 200.0  # 節約目標

class Receipt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    store_name = db.Column(db.String(100), nullable=False)
    purchase_date = db.Column(db.Date, nullable=False)
    items = db.relationship('Item', backref='receipt', lazy=True)

class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    label = db.Column(db.String(50), nullable=True)
    receipt_id = db.Column(db.Integer, db.ForeignKey('receipt.id'), nullable=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    global monthly_budget, savings_goal

    if request.method == 'POST':
        # フォームから予算と節約目標を受け取る
        try:
            monthly_budget = float(request.form.get('monthly_budget', monthly_budget))
            savings_goal = float(request.form.get('savings_goal', savings_goal))
        except ValueError:
            pass  # 何もしない

    receipts = Receipt.query.all()
    items_by_date = defaultdict(list)
    weekly_total = defaultdict(float)
    monthly_total = defaultdict(float)
    monthly_self_invest = defaultdict(float)  # 月ごとの自己投資合計

    for receipt in receipts:
        for item in receipt.items:
            items_by_date[receipt.purchase_date].append(item)
            year, week, _ = receipt.purchase_date.isocalendar()
            week_key = f"{year}-W{week}"
            weekly_total[week_key] += item.amount
            month_key = receipt.purchase_date.strftime('%Y-%m')
            monthly_total[month_key] += item.amount
            # 自己投資の判定（labelが"自己投資"なら加算）
            if item.label == "自己投資":
                monthly_self_invest[month_key] += item.amount

    # 全自己投資の合計（全期間）
    total_self_investment = sum(monthly_self_invest.values())

    return render_template('index.html',
                           items_by_date=dict(items_by_date),
                           weekly_total=dict(weekly_total),
                           monthly_total=dict(monthly_total),
                           monthly_self_invest=dict(monthly_self_invest),
                           receipts=receipts,
                           monthly_budget=monthly_budget,
                           savings_goal=savings_goal,
                           total_self_investment=total_self_investment)

@app.route('/add', methods=['GET', 'POST'])
def add_receipt():
    if request.method == 'POST':
        store_name = request.form['store_name']
        purchase_date = datetime.strptime(request.form['purchase_date'], '%Y-%m-%d').date()
        receipt = Receipt(store_name=store_name, purchase_date=purchase_date)
        db.session.add(receipt)
        db.session.flush()
        item_names = request.form.getlist('item_name')
        item_amounts = request.form.getlist('item_amount')
        item_labels = request.form.getlist('item_label')
        for name, amount, label in zip(item_names, item_amounts, item_labels):
            item = Item(name=name, amount=float(amount), label=label, receipt=receipt)
            db.session.add(item)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('add_receipt.html')

@app.route('/delete_receipt/<int:receipt_id>', methods=['POST'])
def delete_receipt(receipt_id):
    receipt = Receipt.query.get_or_404(receipt_id)
    for item in receipt.items:
        db.session.delete(item)
    db.session.delete(receipt)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/delete_item/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    item = Item.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    return redirect(url_for('index'))

def prepare_monthly_data():
    receipts = Receipt.query.all()
    monthly_totals = defaultdict(float)
    for r in receipts:
        month_key = r.purchase_date.strftime('%Y-%m')
        for item in r.items:
            monthly_totals[month_key] += item.amount
    months = sorted(monthly_totals.keys())
    y = [monthly_totals[m] for m in months]
    X = np.array(range(len(months))).reshape(-1, 1)
    y = np.array(y)
    return X, y, months

@app.route('/analysis')
def analysis():
    X, y, months = prepare_monthly_data()
    if len(X) < 2:
        return render_template('analysis.html',
                               image_base64=None,
                               advice="データが少なすぎて予測できません。",
                               predicted_next=None,
                               budget=monthly_budget)

    model = LinearRegression()
    model.fit(X, y)
    next_month_index = np.array([[X[-1][0] + 1]])
    predicted_next = model.predict(next_month_index)[0]

    latest_month = months[-1]
    label_totals = defaultdict(float)
    items = Item.query.join(Receipt).filter(Receipt.purchase_date.like(f"{latest_month}%")).all()
    for item in items:
        label = item.label or "未分類"
        label_totals[label] += item.amount

    labels = list(label_totals.keys())
    values = list(label_totals.values())

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%')
    ax.set_title(f"{latest_month}の支出内訳")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    diff = predicted_next - monthly_budget
    if diff > 0:
        advice = f"来月の予測支出は ${predicted_next:.2f} で、予算を ${diff:.2f} 超過する可能性があります。節約を検討してください。"
    else:
        advice = f"来月の予測支出は ${predicted_next:.2f} で、予算内に収まる見込みです。"

    return render_template('analysis.html',
                           image_base64=image_base64,
                           advice=advice,
                           predicted_next=predicted_next,
                           budget=monthly_budget)


@app.route('/budget', methods=['GET', 'POST'])
def budget():
    if request.method == 'POST':
        # フォームから送信された予算や節約目標の値を処理する
        monthly_budget = float(request.form.get('monthly_budget', 0))
        saving_goal = float(request.form.get('saving_goal', 0))
        # ここでセッションやDBに保存など必要に応じて処理する
        # 処理後はリダイレクトなど
        return redirect(url_for('index'))
    else:
        # GETならフォームを表示するためのレンダリングなど
        # 予算や節約目標の現在値を渡す（適宜用意）
        monthly_budget = 1000
        saving_goal = 200
        return render_template('budget.html', monthly_budget=monthly_budget, saving_goal=saving_goal)
    
@app.route('/clustering')
def clustering():
    X, y, months = prepare_monthly_data()
    if len(X) < 3:
        return "クラスタリングに十分なデータがありません"

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(y.reshape(-1, 1))

    fig, ax = plt.subplots()
    ax.scatter(months, y, c=kmeans.labels_)
    ax.set_title('支出クラスタリング')
    ax.set_ylabel('月別支出')
    ax.set_xticklabels(months, rotation=45)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return render_template('clustering.html', image_base64=image_base64)

@app.route('/correlation')
def correlation():
    receipts = Receipt.query.all()
    monthly_total = defaultdict(float)
    monthly_self_invest = defaultdict(float)

    for r in receipts:
        key = r.purchase_date.strftime('%Y-%m')
        for item in r.items:
            monthly_total[key] += item.amount
            if item.label == "自己投資":
                monthly_self_invest[key] += item.amount

    common_months = sorted(set(monthly_total.keys()) & set(monthly_self_invest.keys()))
    if len(common_months) < 2:
        return "相関分析に必要な月が足りません"

    x = [monthly_total[m] for m in common_months]
    y = [monthly_self_invest[m] for m in common_months]
    corr, _ = pearsonr(x, y)

    return render_template('correlation.html', corr=round(corr, 3))

@app.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルが選択されていません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイル名が空です')
            return redirect(request.url)
        df = pd.read_csv(file)
        if 'amount' not in df.columns:
            return "CSVに 'amount' 列が必要です"

        mean = df['amount'].mean()
        median = df['amount'].median()
        std = df['amount'].std()

        return render_template('upload_result.html', mean=mean, median=median, std=std)

    return render_template('upload_csv.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
