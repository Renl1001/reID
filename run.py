from app import app
from datetime import timedelta

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
if __name__ == "__main__":
    app.run(debug = True)
