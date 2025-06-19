#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAIクライアント

OpenAI APIを使用してGPTモデルに問い合わせを行うクライアントクラス
"""

import time
import json
import logging
import asyncio
from typing import Dict, List, Optional, Union, Any

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from config import OPENAI_API_KEY, DEFAULT_MODEL, MAX_RETRIES, RETRY_DELAY, BACKOFF_FACTOR, DEBUG

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    OpenAIクライアントクラス
    
    OpenAI APIを使用してGPTモデルに問い合わせを行うクライアントクラス
    エラー時の再問合せ機能を実装
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
        retry_delay: int = RETRY_DELAY,
        backoff_factor: int = BACKOFF_FACTOR
    ):
        """
        初期化
        
        Args:
            api_key: OpenAI APIキー（Noneの場合はconfig.pyから読み込む）
            model: 使用するモデル（Noneの場合はconfig.pyから読み込む）
            max_retries: 最大リトライ回数
            retry_delay: 初期リトライ間隔（秒）
            backoff_factor: バックオフ係数
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("APIキーが設定されていません。環境変数OPENAI_API_KEYを設定するか、初期化時にapi_keyを指定してください。")
        
        self.model = model or DEFAULT_MODEL
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        
        # OpenAIクライアントの初期化（同期版）
        self.client = OpenAI(api_key=self.api_key)
        
        # OpenAIクライアントの初期化（非同期版）
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        logger.debug(f"OpenAIClientを初期化しました。モデル: {self.model}")
    
    def ask(
        self, 
        prompt: str, 
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ChatCompletion:
        """
        GPTモデルに問い合わせを行う
        
        Args:
            prompt: プロンプト
            system: システムプロンプト
            max_tokens: 最大トークン数
            temperature: 温度
            messages: メッセージ履歴（指定した場合はpromptとsystemは無視される）
            **kwargs: その他のパラメータ
            
        Returns:
            ChatCompletion: レスポンス
        """
        retry_count = 0
        last_error = None
        current_delay = self.retry_delay
        
        # メッセージの準備
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        
        while retry_count <= self.max_retries:
            try:
                logger.debug(f"GPTモデルに問い合わせを行います。リトライ回数: {retry_count}")
                
                # パラメータの準備
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                # 最大トークン数が指定されている場合は追加
                if max_tokens:
                    params["max_tokens"] = max_tokens
                
                # その他のパラメータを追加
                for key, value in kwargs.items():
                    params[key] = value
                
                # 問い合わせを実行
                start_time = time.time()
                response = self.client.chat.completions.create(**params)
                end_time = time.time()
                
                logger.debug(f"問い合わせが成功しました。所要時間: {end_time - start_time:.2f}秒")
                return response
                
            except Exception as e:
                # エラーの場合
                logger.warning(f"APIエラーが発生しました: {e}")
                last_error = e
                
                if retry_count < self.max_retries:
                    logger.info(f"{current_delay}秒後にリトライします。({retry_count+1}/{self.max_retries})")
                    time.sleep(current_delay)
                    current_delay *= self.backoff_factor  # バックオフ
                else:
                    logger.error(f"最大リトライ回数（{self.max_retries}回）を超えました。")
                    break
            
            retry_count += 1
        
        # 最大リトライ回数を超えた場合
        if last_error:
            raise last_error
        else:
            raise Exception("不明なエラーが発生しました。")
    
    def extract_text(self, response: ChatCompletion) -> str:
        """
        レスポンスからテキストを抽出する
        
        Args:
            response: レスポンス
            
        Returns:
            str: 抽出されたテキスト
        """
        if not response or not response.choices:
            return ""
        
        return response.choices[0].message.content or ""
    
    def extract_json(self, response: ChatCompletion) -> Dict[str, Any]:
        """
        レスポンスからJSONを抽出する
        
        Args:
            response: レスポンス
            
        Returns:
            Dict[str, Any]: 抽出されたJSON
        """
        text = self.extract_text(response)
        
        # JSONを抽出
        try:
            # JSONブロックを探す
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.warning("JSONが見つかりませんでした。")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSONのデコードに失敗しました: {e}")
            return {}
        except Exception as e:
            logger.error(f"JSONの抽出中に予期しないエラーが発生しました: {e}")
            return {}
    
    def get_usage(self, response: ChatCompletion) -> Dict[str, int]:
        """
        レスポンスから使用量情報を取得する
        
        Args:
            response: レスポンス
            
        Returns:
            Dict[str, int]: 使用量情報
        """
        if not response or not response.usage:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
    async def ask_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ChatCompletion:
        """
        GPTモデルに非同期で問い合わせを行う
        
        Args:
            prompt: プロンプト
            system: システムプロンプト
            max_tokens: 最大トークン数
            temperature: 温度
            messages: メッセージ履歴（指定した場合はpromptとsystemは無視される）
            **kwargs: その他のパラメータ
            
        Returns:
            ChatCompletion: レスポンス
        """
        retry_count = 0
        last_error = None
        current_delay = self.retry_delay
        
        # メッセージの準備
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        
        while retry_count <= self.max_retries:
            try:
                logger.debug(f"GPTモデルに問い合わせを行います。リトライ回数: {retry_count}")
                
                # パラメータの準備
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                # 最大トークン数が指定されている場合は追加
                if max_tokens:
                    params["max_tokens"] = max_tokens
                
                # その他のパラメータを追加
                for key, value in kwargs.items():
                    params[key] = value
                
                # 問い合わせを実行
                start_time = time.time()
                response = await self.async_client.chat.completions.create(**params)
                end_time = time.time()
                
                logger.debug(f"問い合わせが成功しました。所要時間: {end_time - start_time:.2f}秒")
                return response
                
            except Exception as e:
                # エラーの場合
                logger.warning(f"APIエラーが発生しました: {e}")
                last_error = e
                
                if retry_count < self.max_retries:
                    logger.info(f"{current_delay}秒後にリトライします。({retry_count+1}/{self.max_retries})")
                    await asyncio.sleep(current_delay)
                    current_delay *= self.backoff_factor  # バックオフ
                else:
                    logger.error(f"最大リトライ回数（{self.max_retries}回）を超えました。")
                    break
            
            retry_count += 1
        
        # 最大リトライ回数を超えた場合
        if last_error:
            raise last_error
        else:
            raise Exception("不明なエラーが発生しました。")
    
    async def ask_batch_async(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        **kwargs
    ) -> List[ChatCompletion]:
        """
        複数のプロンプトを非同期でバッチ処理する
        
        Args:
            prompts: プロンプトリスト
            system: システムプロンプト
            max_tokens: 最大トークン数
            temperature: 温度
            **kwargs: その他のパラメータ
            
        Returns:
            List[ChatCompletion]: レスポンスリスト
        """
        tasks = []
        for prompt in prompts:
            task = self.ask_async(prompt, system, max_tokens, temperature, **kwargs)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)