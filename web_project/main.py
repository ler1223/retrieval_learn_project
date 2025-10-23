from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from predict import Model

app = Flask(__name__)
model = Model()


@app.route("/")
def index():
    query = request.args.get('q', '')
    products = model.predict(str(query))
    return render_template("index.html", products=products, query=query)


@app.route("/product/<int:product_id>")
def product_detail(product_id):
    # debug_info = model.debug_bm25("smart watch")
    # print("Debug info:", debug_info)
    return jsonify({"error": "Product not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)