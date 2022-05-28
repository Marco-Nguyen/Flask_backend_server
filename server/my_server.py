from flask import Flask, request
from flask_cors import CORS, cross_origin

# Khởi tạo Flask server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/add', methods=['POST', 'GET'])
@cross_origin(origin='*')
def add_process():
    a = int(request.args.get('Num1'))
    b = int(request.args.get('Num2'))
    results = a - b
    return "Result: " + str(results)

# Star Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000')
