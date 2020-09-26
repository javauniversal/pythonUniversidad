
from flask import Flask, jsonify, request
import speech_recognition as sr
import pymysql as MySQLdb
import json
from os import path
import os
##from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import tensorflow as tf
import tensorflow_hub as _hub
import tensorflow_datasets as tfds
from googletrans import Translator
import urllib.request
from pydub import AudioSegment

app = Flask(__name__)

db = MySQLdb.connect("167.99.169.62", "ticlio", "Nueva2020.", "universidad")

@app.route('/listar', methods=['POST'])
def ping():

    req_data = request.get_json()

    producto = req_data['producto']
    
    cursor = db.cursor()
    sqlSelect = "SELECT idhistorial, nombre, texto, idproducto, raw_score, raw_class FROM historial WHERE idproducto =%s"
    cursor.execute(sqlSelect, (producto))
    ##result =  cursor.fetchall()

    data_json = []
    header = [i[0] for i in cursor.description]
    data = cursor.fetchall()
    for i in data:
       data_json.append(dict(zip(header, i)))
    
    print(data_json)

    return jsonify(data_json)


@app.route('/save_record', methods=['POST'])
def save_record():

    req_data = request.get_json()

    _nombreArchivo = req_data['nombreArchivo']
    _urlArchivo = req_data['urlArchivo']
    _idProducto = req_data['idProducto']

    local = '/Users/germangarcia/Documents/audio_m/{}'.format(_nombreArchivo)

    urllib.request.urlretrieve(_urlArchivo, local)

    song = AudioSegment.from_file(local)
    song.export(local, format="wav")

    cursor = db.cursor()
    texto = voz2texto(local)
    cursor.execute("INSERT INTO historial(nombre, texto, idproducto, raw_score, raw_class) VALUES (%s,%s,%s, %s, %s)",(_nombreArchivo, texto, str(_idProducto), "", ""))
    db.commit()

    train_ds, test_ds = tfds.load('imdb_reviews', split=['train', 'test'], batch_size=-1, as_supervised=True)

    train_example, train_labels = tfds.as_numpy(train_ds)
    ##test_example, test_labels = tfds.as_numpy(test_ds)

    model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

    hub_layer = _hub.KerasLayer(model, output_shape=[20], input_shape=[], dtype=tf.string, trainable=True)

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    x_val = train_example[:10000]
    partial_x_train = train_example[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    translator = Translator()

    sqlSelect = "SELECT idhistorial, nombre, texto, idproducto FROM historial WHERE nombre =%s"
    cursor.execute(sqlSelect, (_nombreArchivo))
    result = cursor.fetchone()
    
    
    textString = str(result[2])
    text_en = translator.translate(textString, dest='en')
    new_text_en = tf.convert_to_tensor([text_en.text])
    raw_score = model.predict(new_text_en)
    raw_class = model.predict_classes(new_text_en)

    cursor.execute("""
       UPDATE historial
       SET raw_score=%s, raw_class=%s
       WHERE nombre = %s
    """, (str(raw_score[0][0]), str(raw_class[0][0]), _nombreArchivo))

    db.commit()
    print(raw_score)
    print(raw_class)

    cursor.close()
    db.close()

    return jsonify({"estado": True, "mensaje": "Hola", "datos": "HoZla"})


if __name__ == '__name__':
    app.run()


def voz2texto(AUDIO_FILE):
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)  # read the entire audio file                  
            text = r.recognize_google(audio,language="es-CO")
    return text
