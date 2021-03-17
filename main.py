from flask import Flask, jsonify, request, render_template
#from detect import yolo

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(5)
        return render_template('tem.html')
    pass


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
