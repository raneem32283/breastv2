from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import Column, Integer, String, LargeBinary
import cv2
import numpy as np
import joblib
from datetime import datetime
import io
from bcrypt import hashpw, gensalt, checkpw
from jwt import encode as jwt_encode

app = Flask(__name__)

# إعداد CORS
CORS(app,
     origins=[
         "http://localhost:5173",  # Local frontend (Vite dev server)
         "https://breset-cancer.netlify.app"  # Production frontend (Netlify)
     ],
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"]
)

# إعداد قاعدة البيانات
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  # ضع مفتاح سري قوي
db = SQLAlchemy(app)

# نموذج المستخدم
class User(db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password = Column(LargeBinary, nullable=False)  # تخزين كلمة المرور مشفرة

    def __repr__(self):
        return f'<User {self.name}>'

# إنشاء الجداول داخل السياق
with app.app_context():
    db.create_all()

# تحميل النموذج والمحول
clf = joblib.load('random_forest_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

@app.route('/')
def home():
    return jsonify({
        'message': 'Flask API is running.',
        'available_endpoints': ['/predict', '/enhance', '/login', '/api/register']
    })

# Route: تسجيل مستخدم
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No data provided'}), 400

    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not all([name, email, password]):
        return jsonify({'message': 'Missing fields'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already exists'}), 400

    hashed_password = hashpw(password.encode('utf-8'), gensalt())

    new_user = User(name=name, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201

# Route: تسجيل دخول
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No data provided'}), 400

    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if user and checkpw(password.encode('utf-8'), user.password):
        token = jwt_encode({'user_id': user.id}, app.config['SECRET_KEY'], algorithm='HS256')
        return jsonify({'message': 'Login successful', 'token': token, 'name': user.name}), 200
    else:
        return jsonify({'message': 'Invalid email or password'}), 401

# Route: تحليل الصورة
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = preprocess_image(file)

    prediction = clf.predict(img)
    label = label_encoder.inverse_transform(prediction)[0]

    if label.lower() == 'benign':
        overall = 'Normal'
        recommended_action = 'لا يوجد داعي للقلق، راجع طبيبك للفحص الدوري.'
        findings = []
    elif label.lower() == 'malignant':
        overall = 'Suspicious'
        recommended_action = 'يرجى مراجعة الطبيب المختص لإجراء الفحوصات اللازمة.'
        findings = [{
            'id': 1,
            'location': 'المنطقة القريبة من الثدي الأيسر',
            'severity': 'high',
            'confidenceScore': 0.92,
            'description': 'يوجد تورم مشبوه بحاجة لفحص دقيق.'
        }]
    else:
        overall = 'Normal'
        recommended_action = 'لا توجد علامات غير طبيعية.'
        findings = []

    return jsonify({
        'overallAssessment': overall,
        'recommendedAction': recommended_action,
        'findings': findings,
        'date': datetime.utcnow().isoformat() + 'Z'
    })

# معالجة الصورة
def preprocess_image(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    img = img.reshape(1, -1)
    return img

# Route: تحسين الصورة
@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    enhanced = enhance_grayscale_image(file)

    _, buffer = cv2.imencode('.jpg', enhanced)
    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name='enhanced.jpg'
    )

def enhance_grayscale_image(file_stream):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

if __name__ == '__main__':
    app.run(debug=True)
