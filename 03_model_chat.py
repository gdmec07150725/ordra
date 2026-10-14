"""第二课：向 DeepSeek 发送一个问题，并显示完整回答。

运行：./.venv/Scripts/python.exe -X utf8 ./03_model_chat.py "请介绍自己"
不提供问题时，使用默认问题。每次运行只发起一次请求，不保存聊天历史。
"""

import os  # 标准库：读取环境变量。
from pathlib import Path  # 标准库：以对象形式处理文件路径。
import sys  # 标准库：读取终端参数、返回退出码。

from dotenv import load_dotenv  # 第三方库：把 .env 配置加载到环境变量。
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


def main() -> int:
    """返回 0 表示得到完整回答；非 0 表示配置、请求或回答存在问题。"""

    # __file__ 是当前脚本的路径。with_name() 替换文件名，定位旁边的 .env。
    # 这样即使终端在别的目录启动脚本，也能找到本项目的配置文件。
    env_path = Path(__file__).resolve().with_name(".env")
    # override=False：终端中已经设置的同名环境变量优先于 .env。
    # utf-8-sig 同时支持普通 UTF-8 文件和部分 Windows 编辑器添加的 BOM。
    load_dotenv(dotenv_path=env_path, override=False, encoding="utf-8-sig")

    # sys.argv[1:] 取得脚本名之后的全部参数，join() 用空格把它们连接起来。
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = "请用一句话介绍你自己。"

    if not question:
        print("问题不能为空，请在脚本名后输入一个问题。")
        return 1

    # getenv() 取得环境变量；第二个参数是变量不存在时使用的默认值。
    # 密钥只用于创建客户端，不会通过 print 输出。
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip() or "deepseek-flash"
    if not api_key:
        print(f"尚未配置 DEEPSEEK_API_KEY，请在 {env_path} 中填写并保存。")
        return 1

    # 一条消息是 dict，多条消息放在 list 中，与第一课的数据结构对应。
    # system 说明任务要求；user 携带本次问题。当前没有发送订单数据或工具。
    messages = [
        {"role": "system", "content": "你是一位耐心的中文助手，请简洁、清楚地回答问题。"},
        {"role": "user", "content": question},
    ]

    print(f"正在请求 DeepSeek，模型：{model}")
    print(f"你的问题：{question}")

    try:
        # OpenAI 是兼容 SDK 的类名；base_url 决定请求实际发往 DeepSeek。
        # with 会在离开代码块时关闭客户端连接，包括发生异常的情况。
        with OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=30.0,  # 设置网络操作的超时，单位为秒。
            max_retries=0,  # 本课关闭自动重试，一次运行只尝试一次请求。
        ) as client:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,  # 等完整回答返回后再打印。
                max_tokens=512,  # 限制本次输出长度；Token 不等于字符数。
                # extra_body 将 DeepSeek 的扩展参数合并到请求 JSON 中。
                # 官方模型默认开启思考；本课关闭它，先观察普通问答。
                extra_body={"thinking": {"type": "disabled"}},
            )
    except APITimeoutError:
        # 超时异常属于连接异常的一种，因此要先于 APIConnectionError 捕获。
        print("请求超时，请检查网络或稍后手动重试。")
        return 1
    except APIConnectionError:
        print("无法连接 DeepSeek，请检查网络、代理或证书配置。")
        return 1
    except APIStatusError as error:
        # status_code 是 HTTP 状态码；response.text 是服务端返回的原始正文，
        # 通常包含具体的错误 message、type 等信息，直接显示便于排查问题。
        # 如果正文为空，or 会选择右侧的 SDK 异常描述作为补充。
        detail = error.response.text.strip() or str(error)
        # 如果服务端在错误中回显了当前密钥，替换掉它，其余错误内容保留。
        detail = detail.replace(api_key, "[已隐藏 API Key]")
        print(f"DeepSeek API 返回 HTTP {error.status_code}：\n{detail}")
        return 1

    # SDK 已将返回 JSON 转为 Python 对象，所以这里使用点号读取属性。
    # choices 是候选回答列表，[0] 取得第一项。
    if not response.choices:
        print("接口没有返回候选回答，请稍后重试。")
        return 1

    choice = response.choices[0]
    answer = choice.message.content
    if not answer or not answer.strip():
        print(f"接口没有返回可显示的回答，结束原因：{choice.finish_reason}")
        return 1

    print("\n模型回答：")
    print(answer)

    # usage 是本次请求的用量统计，可能不存在，因此先检查。
    if response.usage is not None:
        usage = response.usage
        print(f"\nToken 用量：输入 {usage.prompt_tokens}，输出 {usage.completion_tokens}，合计 {usage.total_tokens}")

    # 有部分文字不代表正常完成；length 例如表示回答被长度限制截断。
    if choice.finish_reason != "stop":
        print(f"本次回答未正常完成，结束原因：{choice.finish_reason}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
