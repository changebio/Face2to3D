# -*- coding: utf-8 -*-
import os
import shutil
import base64
import time
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
olddir = os.path.join(os.path.curdir, 'old_face')
newdir = os.path.join(os.path.curdir, 'new_face')

imgdata=[]

def save_image(image_base64):
    global olddir
    if not os.path.exists(olddir):
        os.mkdir(olddir)
    image = base64.b64decode(image_base64)
    pid = str(int(time.time()))+'.jpg'
    file_name = os.path.join(olddir, pid)
    with open(file_name, 'wb') as f:
        f.write(image)
    return pid



@app.route('/')
def home():
    return render_template("index.html") # homepage.html在templates文件夹下

@app.route('/face_analyze', methods=['POST'])
def face_analyze():
    if request.method == 'POST':
        try:
            #Save picture
            form_data = request.data
            #image_base64 = form_data.decode()
            image_base64 = form_data.decode().split('base64,')[-1].split('\r\n')[0]

            #print(image_base64)
            pid = save_image(image_base64)


            #TODO processing the picture,return inputdir
            pass
            global olddir
            global newdir
            old_filepath = os.path.join(olddir, pid)
            new_filepath = os.path.join(newdir, pid)
            #print(old_filepath)
            #print(new_filepath)
            shutil.copy(old_filepath, newdir)


            #Send processed picture
            with open(new_filepath, 'rb') as f:
                res = base64.b64encode(f.read())
                result = res.decode('utf-8')
                #global imgdata
                #imgdata.append((str(image_base64),str(res)))
                return jsonify({'image':str(result)})
                #return jsonify({'image':str(image_base64)})


        except Exception as e:
            return jsonify({'error': "please check your html format"})
        return jsonify({'complete':'image saved'})


if __name__ == '__main__':
    app.run()
