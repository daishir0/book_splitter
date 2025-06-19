#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BoundaryAdjusterAgent - 境界調整エージェント

分割されたテキストチャンクの境界を調整し、文章の完結性を保証するエージェント
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from openai_client import OpenAIClient
from utils.helpers import clean_text

logger = logging.getLogger(__name__)


@dataclass
class BoundaryIssue:
    """境界問題を表すデータクラス"""
    chunk_index: int
    issue_type: str  # 'incomplete_start', 'incomplete_end', 'fragment'
    description: str
    suggested_fix: str
    confidence: float


class BoundaryAdjusterAgent:
    """
    境界調整エージェント
    
    分割されたテキストチャンクの境界を調整し、文章の完結性を保証する
    """
    
    def __init__(self, openai_client: OpenAIClient):
        """
        初期化
        
        Args:
            openai_client: OpenAIクライアント
        """
        self.client = openai_client
        self.incomplete_patterns = [
            r'^[ため、そして、また、しかし、だが、それで、つまり、すなわち、なお、ただし、もっとも、ところで、さて、では、そこで、こうして、このように、そのため、したがって、ゆえに、よって]',
            r'^[の、が、を、に、へ、と、で、から、より、まで、について、における、に関して、に対して、によって、として、という、といった、などの、などを]',
            r'^[て、で、し、く、き、い、う、る、れ、ろ、ん]',
            r'^[、。！？]'
        ]
        
    def adjust_boundaries(self, chunks: List[Any], original_text: str) -> List[Any]:
        """
        チャンクの境界を調整する
        
        Args:
            chunks: 分割されたテキストチャンクリスト
            original_text: 元のテキスト
            
        Returns:
            List[Any]: 境界調整されたチャンクリスト
        """
        logger.info("境界調整を開始します")
        
        if not chunks:
            return chunks
        
        # 1. 境界問題を検出
        issues = self._detect_boundary_issues(chunks)
        logger.info(f"検出された境界問題: {len(issues)}件")
        
        # 2. 問題を修正
        adjusted_chunks = self._fix_boundary_issues(chunks, issues, original_text)
        
        # 3. 最終検証
        final_chunks = self._final_validation(adjusted_chunks, original_text)
        
        logger.info(f"境界調整完了: {len(final_chunks)}個のチャンク")
        return final_chunks
    
    def _detect_boundary_issues(self, chunks: List[Any]) -> List[BoundaryIssue]:
        """
        境界問題を検出する
        
        Args:
            chunks: チャンクリスト
            
        Returns:
            List[BoundaryIssue]: 検出された問題リスト
        """
        issues = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.content.strip()
            if not content:
                continue
            
            # 不完全な開始を検出
            start_issue = self._check_incomplete_start(content, i)
            if start_issue:
                issues.append(start_issue)
            
            # 不完全な終了を検出
            end_issue = self._check_incomplete_end(content, i)
            if end_issue:
                issues.append(end_issue)
            
            # フラグメント（断片）を検出
            fragment_issue = self._check_fragment(content, i)
            if fragment_issue:
                issues.append(fragment_issue)
        
        return issues
    
    def _check_incomplete_start(self, content: str, chunk_index: int) -> Optional[BoundaryIssue]:
        """
        不完全な開始をチェック
        
        Args:
            content: チャンクの内容
            chunk_index: チャンクのインデックス
            
        Returns:
            Optional[BoundaryIssue]: 検出された問題（なければNone）
        """
        # 接続詞や助詞で始まる場合
        for pattern in self.incomplete_patterns:
            if re.match(pattern, content):
                return BoundaryIssue(
                    chunk_index=chunk_index,
                    issue_type='incomplete_start',
                    description=f"不完全な開始: '{content[:20]}...'",
                    suggested_fix='前のチャンクと結合',
                    confidence=0.9
                )
        
        # 小文字で始まる場合（英語）
        if re.match(r'^[a-z]', content):
            return BoundaryIssue(
                chunk_index=chunk_index,
                issue_type='incomplete_start',
                description=f"小文字開始: '{content[:20]}...'",
                suggested_fix='前のチャンクと結合',
                confidence=0.7
            )
        
        return None
    
    def _check_incomplete_end(self, content: str, chunk_index: int) -> Optional[BoundaryIssue]:
        """
        不完全な終了をチェック
        
        Args:
            content: チャンクの内容
            chunk_index: チャンクのインデックス
            
        Returns:
            Optional[BoundaryIssue]: 検出された問題（なければNone）
        """
        # 文が途中で切れている場合
        if not re.search(r'[。！？]$', content.strip()):
            # 最後の文が完結していない可能性
            lines = content.strip().split('\n')
            last_line = lines[-1].strip() if lines else ''
            
            if last_line and not re.search(r'[。！？]$', last_line):
                return BoundaryIssue(
                    chunk_index=chunk_index,
                    issue_type='incomplete_end',
                    description=f"不完全な終了: '...{last_line}'",
                    suggested_fix='次のチャンクと結合または境界調整',
                    confidence=0.8
                )
        
        return None
    
    def _check_fragment(self, content: str, chunk_index: int) -> Optional[BoundaryIssue]:
        """
        フラグメント（断片）をチェック
        
        Args:
            content: チャンクの内容
            chunk_index: チャンクのインデックス
            
        Returns:
            Optional[BoundaryIssue]: 検出された問題（なければNone）
        """
        # 非常に短いコンテンツ
        if len(content.strip()) < 30:
            return BoundaryIssue(
                chunk_index=chunk_index,
                issue_type='fragment',
                description=f"短すぎるフラグメント: '{content.strip()}'",
                suggested_fix='前後のチャンクと結合',
                confidence=0.9
            )
        
        return None
    
    def _fix_boundary_issues(self, chunks: List[Any], issues: List[BoundaryIssue], 
                           original_text: str) -> List[Any]:
        """
        境界問題を修正する
        
        Args:
            chunks: チャンクリスト
            issues: 問題リスト
            original_text: 元のテキスト
            
        Returns:
            List[Any]: 修正されたチャンクリスト
        """
        if not issues:
            return chunks
        
        # 問題を重要度順にソート
        sorted_issues = sorted(issues, key=lambda x: x.confidence, reverse=True)
        
        adjusted_chunks = chunks.copy()
        
        for issue in sorted_issues:
            if issue.chunk_index >= len(adjusted_chunks):
                continue
            
            if issue.issue_type == 'incomplete_start' and issue.chunk_index > 0:
                # 前のチャンクと結合
                adjusted_chunks = self._merge_with_previous(adjusted_chunks, issue.chunk_index)
            
            elif issue.issue_type == 'fragment':
                # フラグメントを前後のチャンクと結合
                adjusted_chunks = self._merge_fragment(adjusted_chunks, issue.chunk_index)
            
            elif issue.issue_type == 'incomplete_end':
                # AIで境界を調整
                adjusted_chunks = self._adjust_boundary_with_ai(
                    adjusted_chunks, issue.chunk_index, original_text
                )
        
        return adjusted_chunks
    
    def _merge_with_previous(self, chunks: List[Any], chunk_index: int) -> List[Any]:
        """
        指定されたチャンクを前のチャンクと結合
        
        Args:
            chunks: チャンクリスト
            chunk_index: 結合するチャンクのインデックス
            
        Returns:
            List[Any]: 結合後のチャンクリスト
        """
        if chunk_index <= 0 or chunk_index >= len(chunks):
            return chunks
        
        prev_chunk = chunks[chunk_index - 1]
        current_chunk = chunks[chunk_index]
        
        # 内容を結合
        prev_chunk.content = prev_chunk.content.rstrip() + '\n\n' + current_chunk.content.lstrip()
        prev_chunk.end_pos = current_chunk.end_pos
        prev_chunk.word_count = len(prev_chunk.content)
        
        # 現在のチャンクを削除
        new_chunks = chunks[:chunk_index] + chunks[chunk_index + 1:]
        
        logger.info(f"チャンク {chunk_index} を前のチャンクと結合しました")
        return new_chunks
    
    def _merge_fragment(self, chunks: List[Any], chunk_index: int) -> List[Any]:
        """
        フラグメントを前後のチャンクと結合
        
        Args:
            chunks: チャンクリスト
            chunk_index: フラグメントのインデックス
            
        Returns:
            List[Any]: 結合後のチャンクリスト
        """
        if chunk_index >= len(chunks):
            return chunks
        
        fragment = chunks[chunk_index]
        
        # 前のチャンクと結合を優先
        if chunk_index > 0:
            return self._merge_with_previous(chunks, chunk_index)
        
        # 次のチャンクと結合
        elif chunk_index < len(chunks) - 1:
            next_chunk = chunks[chunk_index + 1]
            next_chunk.content = fragment.content.rstrip() + '\n\n' + next_chunk.content.lstrip()
            next_chunk.start_pos = fragment.start_pos
            next_chunk.word_count = len(next_chunk.content)
            
            new_chunks = chunks[:chunk_index] + chunks[chunk_index + 1:]
            logger.info(f"フラグメント {chunk_index} を次のチャンクと結合しました")
            return new_chunks
        
        return chunks
    
    def _adjust_boundary_with_ai(self, chunks: List[Any], chunk_index: int, 
                               original_text: str) -> List[Any]:
        """
        AIを使用して境界を調整
        
        Args:
            chunks: チャンクリスト
            chunk_index: 調整するチャンクのインデックス
            original_text: 元のテキスト
            
        Returns:
            List[Any]: 調整後のチャンクリスト
        """
        if chunk_index >= len(chunks):
            return chunks
        
        current_chunk = chunks[chunk_index]
        
        # 前後のコンテキストを取得
        context_before = chunks[chunk_index - 1].content[-200:] if chunk_index > 0 else ""
        context_after = chunks[chunk_index + 1].content[:200] if chunk_index < len(chunks) - 1 else ""
        
        system_prompt = """あなたは文章編集の専門家です。
与えられたテキストチャンクの境界を調整し、文章が完結するように修正してください。

以下の原則に従ってください：
1. 文章は意味的に完結している必要がある
2. 接続詞や助詞で始まってはいけない
3. 文は適切な句読点で終わる必要がある
4. 前後の文脈との論理的な流れを保つ

修正されたテキストのみを返してください。説明は不要です。"""
        
        user_prompt = f"""以下のテキストチャンクを文章が完結するように調整してください：

前のコンテキスト:
{context_before}

調整対象のチャンク:
{current_chunk.content}

次のコンテキスト:
{context_after}

このチャンクが独立した完結した文章になるように調整してください。"""
        
        try:
            response = self.client.ask(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            adjusted_content = self.client.extract_text(response).strip()
            
            # 調整結果が妥当かチェック
            if len(adjusted_content) > 0 and len(adjusted_content) < len(current_chunk.content) * 2:
                current_chunk.content = adjusted_content
                current_chunk.word_count = len(adjusted_content)
                logger.info(f"チャンク {chunk_index} の境界をAIで調整しました")
            else:
                logger.warning(f"AI調整結果が不適切です。元のコンテンツを保持: {chunk_index}")
            
        except Exception as e:
            logger.error(f"AI境界調整中にエラーが発生しました: {e}")
        
        return chunks
    
    def _final_validation(self, chunks: List[Any], original_text: str) -> List[Any]:
        """
        最終検証を行う
        
        Args:
            chunks: チャンクリスト
            original_text: 元のテキスト
            
        Returns:
            List[Any]: 検証済みチャンクリスト
        """
        logger.info("最終検証を開始します")
        
        validated_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.content.strip()
            
            if not content:
                logger.warning(f"空のチャンクをスキップ: {i}")
                continue
            
            # 文章完結性の最終チェック
            if self._is_complete_text(content):
                validated_chunks.append(chunk)
            else:
                logger.warning(f"不完全な文章を検出: チャンク {i}")
                # 可能であれば修正、不可能であれば前のチャンクと結合
                if validated_chunks:
                    prev_chunk = validated_chunks[-1]
                    prev_chunk.content = prev_chunk.content.rstrip() + '\n\n' + content
                    prev_chunk.word_count = len(prev_chunk.content)
                    logger.info(f"不完全なチャンク {i} を前のチャンクと結合しました")
                else:
                    # 最初のチャンクの場合はそのまま追加
                    validated_chunks.append(chunk)
        
        logger.info(f"最終検証完了: {len(validated_chunks)}個のチャンク")
        return validated_chunks
    
    def _is_complete_text(self, text: str) -> bool:
        """
        テキストが完結した文章かどうかをチェック
        
        Args:
            text: チェックするテキスト
            
        Returns:
            bool: 完結していればTrue
        """
        text = text.strip()
        
        if not text:
            return False
        
        # 不完全な開始パターンをチェック
        for pattern in self.incomplete_patterns:
            if re.match(pattern, text):
                return False
        
        # 適切な終了をチェック
        if not re.search(r'[。！？]', text):
            return False
        
        # 最低限の長さをチェック
        if len(text) < 10:
            return False
        
        return True
    
    def get_adjustment_summary(self, original_chunks: List[Any], 
                             adjusted_chunks: List[Any]) -> Dict[str, Any]:
        """
        調整結果の要約を取得
        
        Args:
            original_chunks: 元のチャンクリスト
            adjusted_chunks: 調整後のチャンクリスト
            
        Returns:
            Dict[str, Any]: 調整要約
        """
        return {
            'original_chunk_count': len(original_chunks),
            'adjusted_chunk_count': len(adjusted_chunks),
            'chunks_merged': len(original_chunks) - len(adjusted_chunks),
            'adjustment_rate': (len(original_chunks) - len(adjusted_chunks)) / len(original_chunks) if original_chunks else 0,
            'total_content_length': sum(len(chunk.content) for chunk in adjusted_chunks),
            'average_chunk_length': sum(len(chunk.content) for chunk in adjusted_chunks) // len(adjusted_chunks) if adjusted_chunks else 0
        }