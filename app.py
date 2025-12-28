from flask import Flask, render_template, request
import joblib
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)


print("Load Model")
model = joblib.load('model_jaki_final.pkl')
vectorizer = joblib.load('vectorizer_jaki_final.pkl')

# Prepare
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()
factory_stop = StopWordRemoverFactory()
stopwords = factory_stop.get_stop_words()

list_negasi = ['tidak', 'bukan', 'jangan',
               'belum', 'tak', 'tiada', 'ga', 'gak']
               
for kata in list_negasi:
    if kata in stopwords:
        stopwords.remove(kata)

stopwords.extend(['yg', 'dg', 'rt', 'rw', 'dgn', 'ny', 'd', 'klo',
                  'kalo', 'amp', 'biar', 'bikin', 'bilang',
                  'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tuh', 'utk', 'ya',
                  'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                  'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', '&', 'yah',
                  'jaki', 'jakarta', 'aplikasi', 'app', 'apk'])
stopwords = set(stopwords)


norm_dict = {

    'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'engga': 'tidak', 'ngga': 'tidak',
    'gsa': 'tidak bisa', 'gabisa': 'tidak bisa', 'kaga': 'tidak',
    'bs': 'bisa', 'bisaa': 'bisa',
    'skrng': 'sekarang', 'skrg': 'sekarang',
    'sdh': 'sudah', 'dh': 'sudah', 'udh': 'sudah',
    'blm': 'belum',
    'bgt': 'banget', 'bangettt': 'banget',
    'sy': 'saya', 'aku': 'saya', 'gw': 'saya', 'gue': 'saya',
    'ad': 'ada', 'mw': 'mau',
    'lola': 'lambat', 'lemot': 'lambat', 'lelet': 'lambat', 'lama': 'lambat',
    'eror': 'rusak', 'error': 'rusak', 'bug': 'rusak', 'hang': 'rusak',
    'bagus': 'bagus', 'bgs': 'bagus', 'mantap': 'bagus', 'keren': 'bagus', 'good': 'bagus',
    'jelek': 'buruk', 'parah': 'buruk',
    'login': 'masuk', 'log': 'masuk', 'masuk': 'masuk', 'sign': 'masuk',
    'verif': 'verifikasi', 'ktp': 'identitas', 'nik': 'identitas',
    'response': 'respon', 'respon': 'respon',
    'update': 'perbarui', 'download': 'unduh',
    'thanks': 'terima kasih', 'makasih': 'terima kasih'
}


def normalisasi_kata(text):
    words = text.split()
    fixed_words = [norm_dict.get(w, w) for w in words]
    return ' '.join(fixed_words)


def clean_text_website(text):
    # 1. Lowercase
    text = str(text).lower()

    # 2. Cleaning 
    text = re.sub(r'http\S+', '', text)            # Hapus URL
    text = re.sub(r'@\w+', '', text)               # Hapus Mention
    text = re.sub(r'#\w+', '', text)               # Hapus Hashtag
    text = re.sub(r'\d+', '', text)                # Hapus Angka
    text = text.translate(str.maketrans(
        '', '', string.punctuation))  # Hapus Tanda Baca
    text = re.sub(r'[^\w\s]', '', text)            # Hapus Emoji

    # 3. Normalisasi (Slang -> Baku)
    text = normalisasi_kata(text)

    # 4. Stopword Removal
    words = text.split()
    words = [w for w in words if w not in stopwords]


    # 5. Stemming 
    text = ' '.join(words)
    text = stemmer.stem(text)

    return text





@app.route('/', methods=['GET', 'POST'])
def index():
    prediksi_label = None
    probabilitas = None
    ulasan_asli = ""

    if request.method == 'POST':
        ulasan_asli = request.form['ulasan']

        
        ulasan_bersih = clean_text_website(ulasan_asli)

        

        X_input = vectorizer.transform([ulasan_bersih])

        
        hasil = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

        prediksi_label = "POSITIF" if hasil == 1 else "NEGATIF"

        
        confidence = max(proba) * 100
        probabilitas = f"{confidence:.2f}%"

    return render_template('index.html',
                           prediksi=prediksi_label,
                           probabilitas=probabilitas,
                           ulasan=ulasan_asli)


if __name__ == '__main__':
    app.run(debug=True, port=8888)
