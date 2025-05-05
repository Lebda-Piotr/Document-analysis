from flask import Flask, render_template, request
import json

app = Flask(__name__)

@app.route('/')
def review_dashboard():
    with open('enhanced_annotations.json') as f:
        entries = json.load(f)
    return render_template('review.html', entries=entries)

@app.route('/update', methods=['POST'])
def update_entry():
    entry_id = request.form['id']
    # Add validation logic here
    return {"status": "success"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)