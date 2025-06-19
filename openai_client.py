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
        
        if not text.strip():
            logger.warning("レスポンステキストが空です。")
            return {}
        
        # JSONを抽出する複数の方法を試行
        json_candidates = []
        
        # 方法1: ```json ブロックを探す
        import re
        json_blocks = re.findall(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        json_candidates.extend(json_blocks)
        
        # 方法2: ``` ブロック内のJSONを探す
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        for block in code_blocks:
            block = block.strip()
            if block.startswith('{') and block.endswith('}'):
                json_candidates.append(block)
        
        # 方法3: 完全なJSONオブジェクトを探す（ネストした括弧を考慮）
        def find_complete_json(text):
            """完全なJSONオブジェクトを見つける"""
            results = []
            i = 0
            while i < len(text):
                if text[i] == '{':
                    brace_count = 0
                    start = i
                    in_string = False
                    escape_next = False
                    
                    for j in range(i, len(text)):
                        char = text[j]
                        
                        if escape_next:
                            escape_next = False
                            continue
                        
                        if char == '\\':
                            escape_next = True
                            continue
                        
                        if char == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = text[start:j+1]
                                    results.append(json_str)
                                    i = j + 1
                                    break
                    else:
                        break
                else:
                    i += 1
            return results
        
        complete_jsons = find_complete_json(text)
        json_candidates.extend(complete_jsons)
        
        # 方法4: 単純な最初と最後の括弧（フォールバック）
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_candidates.append(text[json_start:json_end])
        
        # 各候補を試してパースする
        for i, json_str in enumerate(json_candidates):
            json_str = json_str.strip()
            if not json_str:
                continue
            
            # JSON文字列の前処理（よくある問題を修正）
            json_str = self._preprocess_json_string(json_str)
            
            try:
                parsed_json = json.loads(json_str)
                logger.debug(f"JSON抽出成功（方法{i+1}）")
                return parsed_json
                
            except json.JSONDecodeError as e:
                logger.debug(f"JSON候補{i+1}のパースに失敗: {e}")
                
                # 追加の修正を試行
                try:
                    fixed_json = self._aggressive_json_fix(json_str)
                    parsed_json = json.loads(fixed_json)
                    logger.debug(f"JSON修正後に抽出成功（方法{i+1}）")
                    return parsed_json
                except:
                    pass
                
                # デバッグ用に問題のある部分を表示
                if len(json_str) > 200:
                    logger.debug(f"問題のあるJSON（最初の200文字）: {json_str[:200]}...")
                else:
                    logger.debug(f"問題のあるJSON: {json_str}")
                continue
            except Exception as e:
                logger.debug(f"JSON候補{i+1}の処理中にエラー: {e}")
                continue
        
        # すべての方法が失敗した場合
        logger.error("すべてのJSON抽出方法が失敗しました。")
        logger.debug(f"元のレスポンステキスト（最初の500文字）: {text[:500]}...")
        
        # 空のJSONを返す
        return {}
    
    def _preprocess_json_string(self, json_str: str) -> str:
        """
        JSON文字列の前処理を行い、よくある問題を修正する
        
        Args:
            json_str: 元のJSON文字列
            
        Returns:
            str: 前処理されたJSON文字列
        """
        import re
        
        # 1. 末尾のカンマを削除
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 2. 不正な改行を修正
        json_str = re.sub(r'\n\s*([,}\]])', r'\1', json_str)
        
        # 3. 重複した引用符を修正
        json_str = re.sub(r'"""([^"]*)"', r'"\1"', json_str)  # """text" -> "text"
        json_str = re.sub(r'"([^"]*)""+', r'"\1"', json_str)  # "text"" -> "text"
        
        # 4. 配列の開始記号の修正
        json_str = re.sub(r':\s*"\[', r': [', json_str)  # ": "[" -> ": ["
        json_str = re.sub(r':\s*\["\s*{', r': [{', json_str)  # ": [" {" -> ": [{"
        
        # 5. 引用符で囲まれていない文字列値を修正（慎重に処理）
        def fix_unquoted_values(match):
            key = match.group(1)
            value = match.group(2).strip()
            
            # 既に引用符で囲まれている場合はそのまま
            if value.startswith('"') and value.endswith('"'):
                return match.group(0)
            
            # 数値、boolean、null、配列、オブジェクトの場合はそのまま
            if (value in ['true', 'false', 'null'] or
                re.match(r'^-?\d+(\.\d+)?$', value) or
                value.startswith('[') or value.startswith('{')):
                return match.group(0)
            
            # その他の場合は引用符で囲む
            return f'"{key}": "{value}"'
        
        # パターンマッチングで修正（より慎重に）
        json_str = re.sub(r'"([^"]+)":\s*([^",}\]\n\[{]+)', fix_unquoted_values, json_str)
        
        # 6. 文字列内のエスケープが必要な文字を処理
        def fix_string_content(match):
            content = match.group(1)
            # 既にエスケープされている場合は重複エスケープを避ける
            if '\\' in content:
                return match.group(0)
            
            # 制御文字をエスケープ
            content = content.replace('\n', '\\n')
            content = content.replace('\r', '\\r')
            content = content.replace('\t', '\\t')
            return f'"{content}"'
        
        # 文字列値のみを対象にエスケープ処理
        def escape_newlines(match):
            content = match.group(1)
            content = content.replace(chr(10), "\\n")
            content = content.replace(chr(13), "\\r")
            content = content.replace(chr(9), "\\t")
            return f': "{content}"'
        
        json_str = re.sub(r':\s*"([^"]*[\n\r\t][^"]*)"', escape_newlines, json_str)
        
        return json_str
    
    def _aggressive_json_fix(self, json_str: str) -> str:
        """
        より積極的なJSON修正を行う
        
        Args:
            json_str: 修正対象のJSON文字列
            
        Returns:
            str: 修正されたJSON文字列
        """
        import re
        
        # 1. 重複引用符の問題を修正
        # """text" -> "text"
        json_str = re.sub(r'"{2,}([^"]*)"', r'"\1"', json_str)
        
        # 2. 配列記号の問題を修正
        # "segments": "[" -> "segments": [
        json_str = re.sub(r':\s*"\[', r': [', json_str)
        json_str = re.sub(r':\s*\]"', r': ]', json_str)
        
        # 3. オブジェクト記号の問題を修正
        # "segments": "[" { -> "segments": [ {
        json_str = re.sub(r':\s*"\[\s*"?\s*{', r': [{', json_str)
        
        # 4. 文字列値の前後の不正な引用符を修正
        # "key": ""value"" -> "key": "value"
        json_str = re.sub(r':\s*"+"([^"]*)"+"', r': "\1"', json_str)
        
        # 5. 配列内の要素の修正
        # [ "{ -> [{
        json_str = re.sub(r'\[\s*"{\s*', r'[{', json_str)
        # }" ] -> }]
        json_str = re.sub(r'}\s*"\s*\]', r'}]', json_str)
        
        # 6. 行の途中で切れた文字列を修正
        lines = json_str.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 引用符で始まって終わらない行を検出
            if line.count('"') % 2 == 1 and not line.endswith(',') and not line.endswith('{') and not line.endswith('['):
                # 次の行と結合を試みる
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('"') and not next_line.startswith('}') and not next_line.startswith(']'):
                        line = line + ' ' + next_line
                        lines[i + 1] = ''  # 次の行をスキップ
            
            fixed_lines.append(line)
        
        json_str = '\n'.join(fixed_lines)
        
        # 7. 最終的な構文チェックと修正
        # 不完全な配列やオブジェクトを検出して修正
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # 不足している閉じ括弧を追加
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        
        return json_str
    
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