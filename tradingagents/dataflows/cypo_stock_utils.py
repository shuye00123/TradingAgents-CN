"""
加密货币数据获取工具
提供对接Binance API的数据获取、处理和格式化功能
"""

import pandas as pd
import requests
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')


class BinanceCryptoProvider:
    """Binance加密货币数据提供器"""

    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        初始化Binance数据提供器

        Args:
            api_key: Binance API Key (可选)
            api_secret: Binance API Secret (可选)
        """
        self.base_url = "https://api.binance.com/api/v3"
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Binance API限制较严格
        self.timeout = 30
        self.max_retries = 5
        self.rate_limit_wait = 60
        self.api_key = api_key
        self.api_secret = api_secret

        logger.info(f"💰 Binance加密货币数据提供器初始化完成")

    def _wait_for_rate_limit(self):
        """等待速率限制"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        """
        发送API请求

        Args:
            endpoint: API端点
            params: 请求参数

        Returns:
            Dict: API响应数据
        """
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()

                response = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    headers={"X-MBX-APIKEY": self.api_key} if self.api_key else {}
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning(f"⚠️ 频率限制 (尝试 {attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.rate_limit_wait)
                    continue
                else:
                    logger.error(f"❌ API请求失败: {response.status_code} - {response.text}")
                    return None

            except Exception as e:
                logger.error(f"❌ API请求异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

        logger.error(f"❌ API请求最终失败: {endpoint}")
        return None

    def get_klines(self, symbol: str, interval: str = '1d', start_time: int = None, end_time: int = None, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        获取K线数据

        Args:
            symbol: 交易对 (如: BTCUSDT)
            interval: K线间隔 (1m, 5m, 1h, 1d等)
            start_time: 开始时间 (毫秒时间戳)
            end_time: 结束时间 (毫秒时间戳)
            limit: 返回数据条数 (最大1000)

        Returns:
            DataFrame: K线数据
        """
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000)
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        data = self._make_request('klines', params)

        if data:
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            # 转换数据类型
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

            # 转换时间戳
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            df['symbol'] = symbol
            return df

        return None

    def get_crypto_data(self, symbol: str, interval: str = '1d', start_date: str = None, end_date: str = None) -> \
    Optional[pd.DataFrame]:
        """
        获取加密货币历史数据 (简化接口)
        改进版本：正确处理不同时间间隔的分批获取

        Args:
            symbol: 交易对
            interval: K线间隔 (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame: 加密货币历史数据
        """
        try:
            symbol = symbol.upper()

            # 设置默认日期
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            logger.info(f"💰 获取加密货币数据: {symbol} ({start_date} 到 {end_date}) 间隔: {interval}")

            # 转换日期为时间戳
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            # 计算每个K线间隔对应的毫秒数
            interval_ms = self._get_interval_milliseconds(interval)
            if interval_ms is None:
                logger.error(f"❌ 不支持的K线间隔: {interval}")
                return None

            # 计算总时间范围和每批获取的时间范围
            total_ms = int((end_dt - start_dt).total_seconds() * 1000)
            batch_ms = 1000 * interval_ms * 1000  # 每批获取1000根K线的时间范围

            # 如果总时间范围小于等于单批范围，一次性获取
            if total_ms <= batch_ms:
                return self._get_single_batch(symbol, interval, start_dt, end_dt)

            # 分批次获取数据
            all_data = []
            current_start = start_dt

            while current_start < end_dt:
                current_end = current_start + timedelta(milliseconds=batch_ms)
                if current_end > end_dt:
                    current_end = end_dt

                batch = self._get_single_batch(
                    symbol=symbol,
                    interval=interval,
                    start_dt=current_start,
                    end_dt=current_end
                )

                if batch is None or batch.empty:
                    break

                all_data.append(batch)

                # 更新下一次请求的开始时间 (当前批次的结束时间)
                current_start = current_end

                # 避免无限循环
                if len(batch) < 1000:
                    break

            if all_data:
                data = pd.concat(all_data)
                data = data.drop_duplicates(subset=['open_time'])
                data = data.sort_values('open_time')

                # 过滤超出请求时间范围的数据
                data = data[(data['open_time'] >= start_dt) & (data['open_time'] <= end_dt)]

                logger.info(f"✅ 加密货币数据获取成功: {symbol}, {len(data)}条记录")
                return data
            else:
                logger.warning(f"⚠️ 加密货币数据为空: {symbol}")
                return None

        except Exception as e:
            logger.error(f"❌ 获取加密货币数据失败: {e}")
            return None

    def _get_single_batch(self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> Optional[
        pd.DataFrame]:
        """获取单批次数据"""
        start_time = int(start_dt.timestamp() * 1000)
        end_time = int(end_dt.timestamp() * 1000)

        return self.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )

    def _get_interval_milliseconds(self, interval: str) -> Optional[int]:
        """将K线间隔转换为毫秒数"""
        interval_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000  # 近似值
        }
        return interval_map.get(interval)

    def get_crypto_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取加密货币基本信息

        Args:
            symbol: 交易对

        Returns:
            Dict: 加密货币信息
        """
        try:
            symbol = symbol.upper()

            logger.info(f"💰 获取加密货币信息: {symbol}")

            # 获取交易对信息
            exchange_info = self._make_request('exchangeInfo')

            if exchange_info and 'symbols' in exchange_info:
                for s in exchange_info['symbols']:
                    if s['symbol'] == symbol:
                        return {
                            'symbol': symbol,
                            'base_asset': s['baseAsset'],
                            'quote_asset': s['quoteAsset'],
                            'status': s['status'],
                            'filters': s['filters'],
                            'source': 'binance'
                        }

            # 如果没找到交易对信息，尝试获取24小时行情
            ticker = self._make_request('ticker/24hr', {'symbol': symbol})

            if ticker:
                return {
                    'symbol': symbol,
                    'price': float(ticker['lastPrice']),
                    'price_change': float(ticker['priceChange']),
                    'price_change_pct': float(ticker['priceChangePercent']),
                    'volume': float(ticker['volume']),
                    'source': 'binance'
                }

            return {
                'symbol': symbol,
                'error': 'Symbol not found',
                'source': 'binance'
            }

        except Exception as e:
            logger.error(f"❌ 获取加密货币信息失败: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'source': 'binance'
            }

    def get_real_time_price(self, symbol: str) -> Optional[Dict]:
        """
        获取加密货币实时价格

        Args:
            symbol: 交易对

        Returns:
            Dict: 实时价格信息
        """
        try:
            symbol = symbol.upper()

            self._wait_for_rate_limit()

            ticker = self._make_request('ticker/24hr', {'symbol': symbol})

            if ticker:
                return {
                    'symbol': symbol,
                    'price': float(ticker['lastPrice']),
                    'open': float(ticker['openPrice']),
                    'high': float(ticker['highPrice']),
                    'low': float(ticker['lowPrice']),
                    'volume': float(ticker['volume']),
                    'price_change': float(ticker['priceChange']),
                    'price_change_pct': float(ticker['priceChangePercent']),
                    'timestamp': datetime.fromtimestamp(ticker['closeTime']/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'quote_asset': ticker.get('quoteAsset', 'USDT')
                }
            else:
                return None

        except Exception as e:
            logger.error(f"❌ 获取加密货币实时价格失败: {e}")
            return None

    def get_top_movers(self, limit: int = 10) -> List[Dict]:
        """
        获取24小时涨幅最大的加密货币

        Args:
            limit: 返回数量

        Returns:
            List: 涨幅最大的加密货币列表
        """
        try:
            tickers = self._make_request('ticker/24hr')

            if tickers:
                # 过滤USDT交易对并计算涨跌幅
                usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]

                # 转换为DataFrame便于排序
                df = pd.DataFrame(usdt_pairs)
                df['priceChangePercent'] = df['priceChangePercent'].astype(float)

                # 获取涨幅最大的
                top_gainers = df.nlargest(limit, 'priceChangePercent')

                result = []
                for _, row in top_gainers.iterrows():
                    result.append({
                        'symbol': row['symbol'],
                        'price': float(row['lastPrice']),
                        'price_change_pct': float(row['priceChangePercent']),
                        'volume': float(row['volume'])
                    })

                return result

            return []

        except Exception as e:
            logger.error(f"❌ 获取涨幅榜失败: {e}")
            return []

    def format_crypto_data(self, symbol: str, data: pd.DataFrame, start_date: str, end_date: str) -> str:
        """
        格式化加密货币数据为文本格式

        Args:
            symbol: 交易对
            data: 加密货币数据DataFrame
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            str: 格式化的加密货币数据文本
        """
        if data is None or data.empty:
            return f"❌ 无法获取加密货币 {symbol} 的数据"

        try:
            # 获取加密货币基本信息
            crypto_info = self.get_crypto_info(symbol)
            base_asset = crypto_info.get('base_asset', symbol.replace('USDT', ''))
            quote_asset = crypto_info.get('quote_asset', 'USDT')

            # 计算统计信息
            latest_price = data['close'].iloc[-1]
            price_change = data['close'].iloc[-1] - data['close'].iloc[0]
            price_change_pct = (price_change / data['close'].iloc[0]) * 100

            avg_volume = data['volume'].mean()
            max_price = data['high'].max()
            min_price = data['low'].min()

            # 格式化输出
            formatted_text = f"""
💰 加密货币数据报告
==================

交易对信息:
- 交易对: {symbol}
- 基础资产: {base_asset}
- 计价资产: {quote_asset}
- 交易所: Binance

价格信息:
- 最新价格: {latest_price:.8f} {quote_asset}
- 期间涨跌: {price_change:+.8f} {quote_asset} ({price_change_pct:+.2f}%)
- 期间最高: {max_price:.8f} {quote_asset}
- 期间最低: {min_price:.8f} {quote_asset}

交易信息:
- 数据期间: {start_date} 至 {end_date}
- 数据条数: {len(data)}条
- 平均成交量: {avg_volume:,.2f} {base_asset}

最近5条数据:
"""

            # 添加最近5天的数据
            recent_data = data.tail(5)
            for _, row in recent_data.iterrows():
                date = row['open_time'].strftime('%Y-%m-%d %H:%M')
                formatted_text += f"- {date}: 开盘{row['open']:.8f}, 收盘{row['close']:.8f}, 成交量{row['volume']:,.2f}\n"

            formatted_text += f"\n数据来源: Binance API\n"

            return formatted_text

        except Exception as e:
            logger.error(f"❌ 格式化加密货币数据失败: {e}")
            return f"❌ 加密货币数据格式化失败: {symbol}"


# 全局提供器实例
_binance_provider = None

def get_binance_provider(api_key: str = None, api_secret: str = None) -> BinanceCryptoProvider:
    """获取全局Binance提供器实例"""
    global _binance_provider
    if _binance_provider is None:
        _binance_provider = BinanceCryptoProvider(api_key, api_secret)
    return _binance_provider


def get_crypto_data(symbol: str, interval: str = '1d', start_date: str = None, end_date: str = None) -> str:
    """
    获取加密货币数据的便捷函数

    Args:
        symbol: 交易对
        interval: K线间隔
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        str: 格式化的加密货币数据
    """
    provider = get_binance_provider()
    data = provider.get_crypto_data(symbol, interval, start_date, end_date)
    return provider.format_crypto_data(symbol, data, start_date, end_date)


def get_crypto_info(symbol: str) -> Dict:
    """
    获取加密货币信息的便捷函数

    Args:
        symbol: 交易对

    Returns:
        Dict: 加密货币信息
    """
    provider = get_binance_provider()
    return provider.get_crypto_info(symbol)


def get_top_movers(limit: int = 10) -> List[Dict]:
    """
    获取涨幅榜的便捷函数

    Args:
        limit: 返回数量

    Returns:
        List: 涨幅最大的加密货币列表
    """
    provider = get_binance_provider()
    return provider.get_top_movers(limit)