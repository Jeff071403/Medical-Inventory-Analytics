from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from pymongo import MongoClient

app = Flask(__name__)

DATA_EXPORT_PATH = "data/medicines_export.csv"
#client = MongoClient("mongodb://localhost:27017/")
#db = client["medical_inventory"]
#collection = db["medicines"]

#Find the last 5 records based on medicine_id
#last_5_records = list(collection.find().sort("medicine_id", -1).limit(1))

#if not last_5_records:
#   print("No records found to delete.")
#else:
    # Delete each record
#   for record in last_5_records:
#        collection.delete_one({"_id": record["_id"]})
# Helper: Load and clean dataframe 
def load_cleaned_df():
    if not os.path.exists(DATA_EXPORT_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_EXPORT_PATH)
    #df = df.iloc[:-5]
    
    df.columns = df.columns.str.strip().str.lower()
    
    if "name" in df.columns and "medicine_name" not in df.columns:
        df = df.rename(columns={"name": "medicine_name"})
    
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip().str.lower()
    if "medicine_name" in df.columns:
        df["medicine_name"] = df["medicine_name"].astype(str).str.strip()
    
    for col in ["last_sold_date", "purchase_date", "expiry_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

df_raw = load_cleaned_df()

available_categories = sorted(df_raw['category'].dropna().unique()) if not df_raw.empty else []
available_medicines = sorted(df_raw['medicine_name'].dropna().unique()) if not df_raw.empty else []

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html", chart_data=None, message=None, categories=available_categories)

@app.route("/analyze", methods=["POST"])
def analyze():
    selected_filter = request.form.get('filter')
    df = load_cleaned_df()

    if df.empty:
        return render_template("analytics.html", chart_data=None, message="No data available.", categories=available_categories)

    if selected_filter == "top5":
        top_meds = df.groupby('medicine_name')['quantity'].sum().nlargest(5)
        fig, ax = plt.subplots(figsize=(8, 5))
        top_meds.plot(kind='bar', color='salmon', ax=ax)
        ax.set_title("Top 5 Most Stocked Medicines")
        ax.set_ylabel("Quantity")
        plt.xticks(rotation=45)

    else:
        if selected_filter != 'all':
            df = df[df['category'] == selected_filter]

        if df.empty:
            return render_template("analytics.html", chart_data=None,
                                   message="No data found for selected category.",
                                   categories=available_categories)

        grouped = df.groupby('category').agg({'quantity': 'sum', 'selling_price': 'mean'})
        grouped['total_value'] = grouped['quantity'] * grouped['selling_price']

        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        grouped['quantity'].plot(kind='bar', ax=ax[0], color='skyblue')
        ax[0].set_title("Stock Quantity per Category")
        ax[0].set_ylabel("Quantity")

        grouped['total_value'].plot(kind='bar', ax=ax[1], color='lightgreen')
        ax[1].set_title("Total Value per Category")
        ax[1].set_ylabel("Total Value (₹)")

        plt.tight_layout()

    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
    plt.close()

    return render_template("analytics.html", chart_data=img_base64,
                           message=None, categories=available_categories)

@app.route("/piecharts", methods=["POST"])
def piecharts():
    pie_filter = request.form.get('pie_filter')
    df = load_cleaned_df()

    if df.empty:
        return render_template("analytics.html", chart_data=None, message="No data available.", categories=available_categories)

    if pie_filter == "category_share":
        category_share = df.groupby('category')['quantity'].sum()
        if category_share.empty:
            return render_template("analytics.html", chart_data=None,
                                   message="No category data found.",
                                   categories=available_categories)
        fig, ax = plt.subplots(figsize=(6, 6))
        category_share.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
        ax.set_ylabel("")
        ax.set_title("Category-wise Stock Distribution")

    elif pie_filter == "manufacturer_share":
        if 'manufacturer' not in df.columns:
            return render_template("analytics.html", chart_data=None,
                                   message="No manufacturer data found.",
                                   categories=available_categories)
        manufacturer_share = df.groupby('manufacturer')['quantity'].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(7, 7))
        manufacturer_share.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
        ax.set_ylabel("")
        ax.set_title("Manufacturer Market Share (Top 10)")

    else:
        return render_template("analytics.html", chart_data=None,
                               message="Invalid pie chart option.",
                               categories=available_categories)

    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
    plt.close()

    return render_template("analytics.html", chart_data=img_base64,
                           message=None, categories=available_categories)

@app.route("/trends", methods=["POST"])
def trends():
    trend_type = request.form.get('trend_filter')
    df = load_cleaned_df()

    if df.empty:
        return render_template("analytics.html", chart_data=None, message="No data available.", categories=available_categories)

    if trend_type == "sales":
        if df['last_sold_date'].isna().all():
            return render_template("analytics.html", chart_data=None, message="No sales date data found.", categories=available_categories)

        df['month'] = df['last_sold_date'].dt.to_period('M')
        sales_by_month = df.groupby('month')['total_sold'].sum()

        fig, ax = plt.subplots(figsize=(8, 5))
        sales_by_month.plot(kind='line', marker='o', color='orange', ax=ax)
        ax.set_title("Sales Trend Over Months")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Units Sold")
        plt.xticks(rotation=45)

    elif trend_type == "purchases":
        if df['purchase_date'].isna().all():
            return render_template("analytics.html", chart_data=None, message="No purchase date data found.", categories=available_categories)

        df['month'] = df['purchase_date'].dt.to_period('M')
        purchases_by_month = df.groupby('month')['quantity'].sum()

        fig, ax = plt.subplots(figsize=(8, 5))
        purchases_by_month.plot(kind='line', marker='o', color='blue', ax=ax)
        ax.set_title("Purchase Trend Over Months")
        ax.set_xlabel("Month")
        ax.set_ylabel("Quantity Purchased")
        plt.xticks(rotation=45)

    else:
        return render_template("analytics.html", chart_data=None, message="Invalid trend type selected.", categories=available_categories)

    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
    plt.close()

    return render_template("analytics.html", chart_data=img_base64, message=None, categories=available_categories)

@app.route("/intensity", methods=["POST"])
def intensity():
    intensity_filter = request.form.get('intensity_filter')
    df = load_cleaned_df()

    if df.empty:
        return render_template("analytics.html", chart_data=None, message="No data available.", categories=available_categories)

    if intensity_filter == "category_sales":
        pivot = df.pivot_table(values="total_sold", index="category", aggfunc="sum").fillna(0)
        pivot_norm = (pivot - pivot.min()) / (pivot.max() - pivot.min())

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_norm, annot=True, cmap="YlOrRd", fmt=".2f", ax=ax)
        ax.set_title("Category vs. Sales Density (Normalized 0–1)")

    elif intensity_filter == "vendor_supply":
        if 'vendor_name' not in df.columns:
            return render_template("analytics.html", chart_data=None,
                                   message="Vendor data not available.",
                                   categories=available_categories)

        pivot = df.pivot_table(values="quantity", index="vendor_name", aggfunc="sum").fillna(0).nlargest(20, "quantity")
        pivot_norm = (pivot - pivot.min()) / (pivot.max() - pivot.min())

        fig, ax = plt.subplots(figsize=(8, 10))
        sns.heatmap(pivot_norm, annot=True, cmap="Blues", fmt=".2f", ax=ax)
        ax.set_title("Vendor vs. Supply Density (Top 20, Normalized 0–1)")

    else:
        return render_template("analytics.html", chart_data=None,
                               message="Invalid intensity option.",
                               categories=available_categories)

    img_io = io.BytesIO()
    plt.savefig(img_io, format="png", bbox_inches="tight")
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode("utf-8")
    plt.close()

    return render_template("analytics.html", chart_data=img_base64,
                           message=None, categories=available_categories)

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    df = load_cleaned_df()
    results = []

    future_days_choice = int(request.form.get("future_days", 30)) if request.method == "POST" else 30  

    for med in available_medicines:
        med_df = df[df['medicine_name'] == med].copy()

        med_df['quantity'] = pd.to_numeric(med_df['quantity'], errors='coerce').fillna(0)
        med_df['total_sold'] = pd.to_numeric(med_df['total_sold'], errors='coerce').fillna(0)

        if len(med_df) < 2:
            current_stock = med_df['quantity'].mean() if not med_df.empty else 0
            threshold = med_df['min_stock_threshold'].mean() if not med_df.empty else 0
            status = "Insufficient data"
            y_future = current_stock

            results.append({
                "medicine": med,
                "current_stock": round(current_stock, 2),
                "predicted_stock": round(y_future, 2),
                "min_threshold": round(threshold, 2),
                "status": status
            })
            continue

        med_df['days_since_purchase'] = (pd.to_datetime("today") - med_df['purchase_date']).dt.days
        X = med_df[['days_since_purchase']]
        y = med_df['quantity']

        model = LinearRegression()
        model.fit(X, y)

        future_days = med_df['days_since_purchase'].max() + future_days_choice
        y_future = model.predict([[future_days]])[0]

        threshold = med_df['min_stock_threshold'].mean()
        current_stock = med_df['quantity'].mean()

        if y_future <= 0:
            status = "Out of Stock Soon"
        elif y_future < threshold:
            status = "At Risk"
        else:
            status = "Safe"

        results.append({
            "medicine": med,
            "current_stock": round(current_stock, 2),
            "predicted_stock": round(y_future, 2),
            "min_threshold": round(threshold, 2),
            "status": status
        })

    return render_template("prediction.html", results=results, future_days=future_days_choice)

@app.route("/demand", methods=["GET", "POST"])
def demand_prediction():
    df = df_raw.copy()
    results = []

    future_days_choice = int(request.form.get("future_days", 30)) if request.method == "POST" else 30  

    for med in available_medicines:
        med_df = df[df['medicine_name'] == med].copy()

        med_df['total_sold'] = pd.to_numeric(med_df['total_sold'], errors='coerce').fillna(0)
        med_df['quantity'] = pd.to_numeric(med_df['quantity'], errors='coerce').fillna(0)
        med_df['selling_price'] = pd.to_numeric(med_df['selling_price'], errors='coerce').fillna(0)
        med_df['days_since_purchase'] = (pd.to_datetime("today") - med_df['purchase_date']).dt.days

        if len(med_df) < 2:
            avg_demand = med_df['total_sold'].mean() if not med_df.empty else 0
            results.append({
                "medicine": med,
                "predicted_demand": round(avg_demand, 2),
                "future_days": future_days_choice,
                "status": "Insufficient data"
            })
            continue

        features = ['days_since_purchase', 'quantity', 'selling_price', 'category', 'vendor_name']
        X = pd.get_dummies(med_df[features], drop_first=True)
        y = med_df['total_sold']

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        future_data = {
            'days_since_purchase': med_df['days_since_purchase'].max() + future_days_choice,
            'quantity': med_df['quantity'].iloc[-1],
            'selling_price': med_df['selling_price'].iloc[-1],
            'category': med_df['category'].iloc[-1],
            'vendor_name': med_df['vendor_name'].iloc[-1]
        }

        future_df = pd.DataFrame([future_data])
        future_X = pd.get_dummies(future_df[features], drop_first=True)
        future_X = future_X.reindex(columns=X.columns, fill_value=0)
        future_X_poly = poly.transform(future_X)

        y_future = max(model.predict(future_X_poly)[0], 0)

        results.append({
            "medicine": med,
            "predicted_demand": round(y_future, 2),
            "future_days": future_days_choice,
        })

    return render_template("demand.html", results=results, future_days=future_days_choice)

@app.route("/add", methods=["GET", "POST"])
def add():
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["medical_inventory"]
    collection = db["medicines"]

    
    last_record = collection.find_one(sort=[("medicine_id", -1)])
    next_id = int(last_record["medicine_id"]) + 1 if last_record and last_record.get("medicine_id") is not None else 1

    if request.method == "POST":
        new_medicine = {
            "medicine_id": next_id,
            "name": request.form.get("name"),
            "category": request.form.get("category"),
            "manufacturer": request.form.get("manufacturer"),
            "batch_number": request.form.get("batch_number"),
            "purchase_date": request.form.get("purchase_date"),
            "expiry_date": request.form.get("expiry_date"),
            "quantity": request.form.get("quantity"),
            "purchase_price": request.form.get("purchase_price"),
            "selling_price": request.form.get("selling_price"),
            "vendor_name": request.form.get("vendor_name"),  
            "min_stock_threshold": request.form.get("min_stock_threshold"),
            "total_sold": request.form.get("total_sold"),
            "last_sold_date": request.form.get("last_sold_date"),
            "status": request.form.get("status")
        }

        
        collection.insert_one(new_medicine)
        next_id += 1

    
    unique_categories = collection.distinct("category")
    unique_medicines = collection.distinct("name")
    unique_manufacturers = collection.distinct("manufacturer")
    unique_vendors = collection.distinct("vendor_name")

    return render_template(
        "add.html",
        categories=unique_categories,
        medicines=unique_medicines,
        manufacturers=unique_manufacturers,
        vendors=unique_vendors,
        next_id=next_id
    )



@app.route("/export", methods=["GET"])
def export_to_csv():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["medical_inventory"]
    collection = db["medicines"]

    data = list(collection.find({}, {"_id": 0}))  

    if not data:
        return "No data available to export."

    df = pd.DataFrame(data)
    df.to_csv(DATA_EXPORT_PATH, index=False)

    return f"Data exported successfully to {DATA_EXPORT_PATH}"

@app.route("/vendor_manufacturer")
def vendor_manufacturer_insights():
    df = load_cleaned_df()

    if df.empty:
        return render_template("vendor_manufacturer.html", vendor_supply=[], manufacturer_share=[])

    if "vendor_name" in df.columns and "quantity" in df.columns:
        vendor_supply = (
            df.groupby("vendor_name")["quantity"]
            .sum()
            .reset_index()
            .sort_values(by="quantity", ascending=False)
            .to_dict(orient="records")
        )
    else:
        vendor_supply = []

    if "manufacturer" in df.columns and "total_sold" in df.columns:
        manufacturer_share = (
            df.groupby("manufacturer")["total_sold"]
            .sum()
            .reset_index()
            .sort_values(by="total_sold", ascending=False)
            .to_dict(orient="records")
        )
    else:
        manufacturer_share = []

    return render_template(
        "vendor_manufacturer.html",
        vendor_supply=vendor_supply,
        manufacturer_share=manufacturer_share
    )
import pandas as pd
from flask import send_file

@app.route("/export_csv")
def export_csv():
    results = [
        {"medicine": "Paracetamol", "current_stock": 50, "predicted_stock": 30, "min_threshold": 20, "status": "Safe"},
        {"medicine": "Ibuprofen", "current_stock": 40, "predicted_stock": 10, "min_threshold": 15, "status": "At Risk"},
    ]

    # Save CSV
    file_path = "data/stockout_predictions.csv"
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)

    # Send CSV
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
