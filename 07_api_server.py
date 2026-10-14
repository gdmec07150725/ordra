"""第六、七课入口：启动后打开 http://127.0.0.1:8000。"""

import uvicorn


if __name__ == "__main__":
    # api_app:app 表示导入 api_app.py 中名为 app 的 FastAPI 对象。
    # 第七课把历史保存在 SQLite；busy 和缓存仍在内存，继续使用单进程。
    uvicorn.run("api_app:app", host="127.0.0.1", port=8000)
