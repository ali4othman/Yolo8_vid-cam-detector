from flask import Flask,redirect,render_template,request,Response,url_for,session,after_this_request
from ultralytics import YOLO
import cv2
import math
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
import uuid
from flask_wtf.file import FileAllowed,FileRequired
from wtforms import StringField,FileField,SubmitField,IntegerRangeField
from wtforms.validators import Length,InputRequired,NumberRange
import os

app = Flask(__name__,template_folder='templates',static_folder='static',static_url_path='/')

app.config['SECRET_KEY'] = 'PLHMFfl'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100 mb

class Upload(FlaskForm):
    file = FileField("Upload video",validators=[FileRequired(message="this field is required!"),
                                        FileAllowed(['mp4','mov','avi'],message="only vids!")]) # uploaded vid stored here
    submit = SubmitField("Run")

model = YOLO('yolov8s.pt')
def yolo_detect(path):
    cap = cv2.VideoCapture(path)
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        results = model(frame,stream=True)
        for res in results:
            boxes = res.boxes
            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),thickness=3)
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                cls = model.names[cls]
                label = f'{cls}{conf}'
                t_size = cv2.getTextSize(label,0,1,2)[0] # (w,h), baseline
                c2 = x1 + t_size[0],y1 - t_size[1] - 3 # left, top
                cv2.rectangle(frame,(x1,y1),c2,(255,0,255),thickness=-1,lineType=cv2.LINE_AA)
                cv2.putText(frame,label,(x1,y1-3),0,1,(255,255,255),1,lineType=cv2.LINE_AA) # bottom left
        yield frame
    cv2.destroyAllWindows()

def generate_frames(path):
    frames = yolo_detect(path)
    for frame in frames:
        ref,buffer = cv2.imencode('.jpg',frame) # 1D byte narray uint8
        frame = buffer.tobytes()
        # frame is bytes so make sure to use b prefix with all concatenated strings
        yield(b'--frame\r\n' # b => bytes || --anything used to split frames
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # --word this word will be boundary in Response mimetype

@app.route('/',methods=['GET','POST'])
def main():
    form = Upload()
    if form.validate_on_submit():
        session.clear()
        vid = form.file.data
        special_code = uuid.uuid4().hex[:4]
        f, exe = os.path.splitext(secure_filename(vid.filename))
        os.makedirs(os.path.join(os.getcwd(),'uploads'), exist_ok=True)
        vid.save(os.path.join(os.path.abspath(os.path.dirname(__name__)),app.config['UPLOAD_FOLDER'],
                              f'{f}{special_code}{exe}'))
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__name__)),app.config['UPLOAD_FOLDER'],f'{f}{special_code}{exe}')
        return redirect(url_for('video'))
    return render_template('index.html',form=form)

@app.route('/video')
def video():
    return Response(generate_frames(path=session.get('video_path',None)),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/webcam')
def webcam():
    return Response(generate_frames(path=0),mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    app.run(debug=True)
