#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
書籍構造化分割ツール設定ファイル
"""

import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIキー
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# デフォルトのモデル
DEFAULT_MODEL = "gpt-4o-mini"

# リトライ設定
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒
BACKOFF_FACTOR = 2  # バックオフ係数

# ログ設定
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# 出力設定
MAX_TOKENS = 2000
TEMPERATURE = 0.3  # 構造化タスクなので低めに設定