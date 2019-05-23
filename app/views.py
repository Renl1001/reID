from app import app
from flask import Flask, render_template, request, url_for
import json
import os
import pickle
import posixpath
import time
from app.utils import recognition
from werkzeug import secure_filename
from app.pagination import Pagination
import pandas as pd

# 用户上传图片的保存路径
SAVE_DIR = "app/static/images/unknown"
# 允许用户上传的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# app = Flask(__name__)
app.config['MAX_CONTEN T_LENGTH'] = 5 * 1024 * 1024

def get_current_time():
    return str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())))

# 用户访问根目录时的路由行为
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# 用户访问人脸库时的路由行为
@app.route('/personlib')
def personlib():
    df = pd.read_csv(os.path.join('app', 'data', 'g_data.csv'))
    g_path = df['g_path'].tolist()
    pager_obj = Pagination(request.args.get("page", 1), len(g_path), request.path, request.args, per_page_count=18)
    print(request.path)
    print(request.args)
    index_list = g_path[pager_obj.start:pager_obj.end]
    html = pager_obj.page_html()
    # return render_template("test.html", index_list=index_list, )
    return render_template('personlib.html', image_infos=index_list, html=html)

# 用户访问人脸识别模块时的路由行为
@app.route('/personrcg', methods=['GET', 'POST'])
def personrcg():
    # 如果用户提交POST表单
    if request.method == 'POST':

        # 记录时间开销
        timecosts = []

        # 用户提出POST开始计时
        time_post_start = time.time()
        print("[ ]开始处理POST请求")

        # POST表单中没有文件时 提示用户先选择图片
        if 'file' not in request.files:
            return render_template('personrcg.html', unknown_show=True,
                                   similarity=0, result="unknown",
                                   message="请先选择图片")

        # 读取表单内的文件
        file = request.files['file']

        # 文件合法
        if file and allowed_file(file.filename):
            # 使用安全文件名以避免中文等字符的出现
            file_type = secure_filename(file.filename).split('.')[-1]
            file_name = get_current_time() + '.' + file_type
            print(file_name)

            #  用户上传图片的保存路径
            image_path = posixpath.join(SAVE_DIR, file_name)

            # 文件名重复时更新文件名
            cnt = 1
            while os.path.exists(image_path):
                basename, filetype = file_name.split('.')
                file_name = basename + '_' + str(cnt) + '.' + filetype
                image_path = posixpath.join(SAVE_DIR, file_name)
            file.save(image_path)
            print("[+]保存成功! 保存路径{}".format(image_path))

            # 传输结束 记录传输时间
            time_trans_end = time.time()
            trans_timecost = time_trans_end - time_post_start
            timecosts.append(trans_timecost)

            image_paths = recognition(image_path)
            person_ids, cs, ss, seconds = [],[],[],[]
            datas = []
            result = image_paths[0].split('/')[-1].split('_')[0]
            for path in image_paths:
                wd = path.split('/')[-1]
                wd = wd.split('_')
                person_ids.append(wd[0])
                cs.append(wd[1][0:2])
                ss.append(wd[1][2:])
                # seconds.append(wd[2]/25)
                datas.append((path, wd[0], wd[1][0:2], wd[1][2:], int(wd[2])/25))
            return render_template('personrcg.html', image_paths=image_paths, datas = datas,
                    left_photo_path=os.path.join('static/images/unknown',file_name),
                    result=result, message="识别成功")
        else:# 文件不合法
            print("[-]图片上传有误")
            left_photo_path = url_for('static', filename='images/loading.gif')
            return render_template('personrcg.html', left_photo_path=left_photo_path,
                                   similarity=0, result="unknown", message="图片上传有误")

    # 用户向本页面发出GET请求 返回该页面内容
    else:
        return render_template('personrcg.html', unknown_show=True, similarity=0, result="unknown", message=None)


# 用户访问"关于我们"页面的路由行为
@app.route('/about')
def about():
    return render_template('about.html')

# 运行flask项目
if __name__ == '__main__':
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    # 运行flask主程序
    app.run()
