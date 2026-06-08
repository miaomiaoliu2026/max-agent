import os
import subprocess

from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


@tool
def get_weather(location: str) -> str:
    """
    获取指定城市的天气信息

    Args:
        location: 城市名称，例如 "Beijing", "Shanghai"

    Returns:
        天气信息的字符串描述
    """
    import requests

    # OpenWeather API 配置
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "错误：未设置 OPENWEATHER_API_KEY 环境变量"

    base_url = "https://api.openweathermap.org/data/2.5/weather"

    # 构建请求参数
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric",  # 使用摄氏度
        "lang": "zh_cn"  # 中文返回
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 解析天气数据
        city = data["name"]
        country = data["sys"]["country"]
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        weather_desc = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]

        result = (
            f"{city} ({country}) 的天气情况：\n"
            f"天气：{weather_desc}\n"
            f"温度：{temperature}°C\n"
            f"体感温度：{feels_like}°C\n"
            f"湿度：{humidity}%\n"
            f"风速：{wind_speed} m/s"
        )
        return result

    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return f"错误：找不到城市 '{location}'"
        elif response.status_code == 401:
            return "错误：API Key 无效"
        else:
            return f"错误：HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return "错误：请求超时"
    except requests.exceptions.RequestException as e:
        return f"错误：请求失败 - {str(e)}"
    except KeyError as e:
        return f"错误：解析天气数据失败 - {str(e)}"


@tool
def execute_code(code: str):
    """
    用于执行大模型输出的python代码
    :param code: 完整的python代码
    :return: 代码输出的结果
    """
    logger.info("\n" + "=" * 60)
    logger.info("⚠️  即将执行以下代码:")
    logger.info("=" * 60)
    logger.info(code)
    logger.info("=" * 60)

    # 询问用户是否继续执行
    # user_input = input("是否执行此代码? (y/n): ").strip().lower()

    # if user_input not in ['y', 'yes']:
    #     logger.info("❌ 用户取消执行")
    #     return "用户取消了代码执行"

    logger.info("✅ 开始执行代码...")
    try:
        with open("temp.py", "w", encoding="utf-8") as f:
            f.write(code)
        run = subprocess.run(['python', 'temp.py'], capture_output=True, text=True, timeout=30, encoding='utf-8')

        # 检查是否有错误
        if run.returncode != 0:
            logger.info(f"❌ 代码执行失败,返回码: {run.returncode}")
            if run.stderr:
                logger.info(f"错误信息: {run.stderr}")
            return f"执行失败:\n{run.stderr}"

        logger.info("✅ 代码执行成功")
        return run.stdout
    except subprocess.TimeoutExpired:
        logger.error("❌ 代码执行超时(超过30秒)")
        return "执行超时:代码运行时间过长"
    except Exception as e:
        logger.error(f"❌ 执行出错: {type(e).__name__} - {str(e)}")
        return f"执行异常: {type(e).__name__} - {str(e)}"
