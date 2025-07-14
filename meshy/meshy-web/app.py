from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

API_KEY = "msy_1u5sWuBnLEG1pO9cTbrIY9VLo3pG7sqLy0AV"
BASE_URL = "https://api.meshy.ai/openapi/v2"  #updated version

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/meshy/text-to-3d', methods=['POST'])
def text_to_3d():
    data = request.get_json()
    prompt = data.get('prompt', 'A cute robot')

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "mode": "preview",
        "prompt": prompt,
        "negative_prompt": "low quality, low resolution, low poly, ugly",
        "art_style": "realistic",
        "should_remesh": True
    }

    try:
        response = requests.post(f"{BASE_URL}/text-to-3d", headers=headers, json=payload)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status_code": response.status_code if 'response' in locals() else None,
            "raw": response.text if 'response' in locals() else None
        })

if __name__ == '__main__':
    app.run(debug=True)
