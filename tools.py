import os
from langchain_core.tools import tool


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
