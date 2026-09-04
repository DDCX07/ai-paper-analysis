# run_app.py
# 应用启动器：先给Streamlit静态服务打补丁，再正常启动应用。
#
# 背景：概念图谱的mermaid与ELK布局引擎自托管在 ./static（避免每次从CDN拉
# 2MB+资源导致加载慢）。Streamlit的AppStaticFileHandler默认只对白名单扩展名
# 返回真实Content-Type，.js/.mjs被强制text/plain+nosniff，浏览器会拒载，
# 因此必须在服务进程里把这两个扩展名加进白名单。
#
# 用法： python run_app.py run app.py --server.port 8501
import mimetypes

import streamlit.web.server.app_static_file_handler as _asfh

if ".js" not in _asfh.SAFE_APP_STATIC_FILE_EXTENSIONS:
    _asfh.SAFE_APP_STATIC_FILE_EXTENSIONS = tuple(
        _asfh.SAFE_APP_STATIC_FILE_EXTENSIONS
    ) + (".js", ".mjs")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

from streamlit.web import cli as stcli

if __name__ == "__main__":
    stcli.main()
