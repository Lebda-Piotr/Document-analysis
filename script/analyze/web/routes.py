from flask import Blueprint, render_template, request, jsonify
import json
import os

review_bp = Blueprint('review', __name__)

@review_bp.route('/')
def review_dashboard():
    data = load_annotations()
    return render_template('review.html', entries=data['entries'])

@review_bp.route('/update', methods=['POST'])
def update_entry():
    entry_id = request.json.get('id')
    correction = request.json.get('correction')
    
    data = load_annotations()
    
    if 0 <= entry_id < len(data['entries']):
        data['entries'][entry_id]['user_correction'] = correction
        data['entries'][entry_id]['needs_review'] = False
        save_annotations(data)
        return jsonify({"status": "success"})
    
    return jsonify({"status": "error", "message": "Invalid entry ID"})

def load_annotations():
    try:
        with open('enhanced_annotations.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"count": 0, "entries": []}

def save_annotations(data):
    with open('enhanced_annotations.json', 'w') as f:
        json.dump(data, f, indent=2)