#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SplitterAgent - 節単位分割エージェント

セグメントに応じて本文を切り出すエージェント
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from openai_client import OpenAIClient
from utils.helpers import split_text_by_positions, clean_text
from agents.segmenter import StructureSegment
from agents.boundary_adjuster import BoundaryAdjusterAgent

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """テキストチャンクを表すデータクラス"""
    type: str  # 'chapter' or 'section'
    number: str
    title: str
    content: str
    start_pos: int
    end_pos: int
    parent_chapter: Optional[int] = None
    word_count: int = 0


class SplitterAgent:
    """
    節単位分割エージェント
    
    構造セグメントに基づいてテキストを適切に分割する
    """
    
    def __init__(self, openai_client: OpenAIClient):
        """
        初期化
        
        Args:
            openai_client: OpenAIクライアント
        """
        self.client = openai_client
        self.chunks = []
        self.boundary_adjuster = BoundaryAdjusterAgent(openai_client)
        
    def split_text(self, text: str, segments: List[StructureSegment]) -> List[TextChunk]:
        """
        テキストをセグメントに基づいて分割する
        
        Args:
            text: 入力テキスト
            segments: 構造セグメントリスト
            
        Returns:
            List[TextChunk]: 分割されたテキストチャンクリスト
        """
        logger.info("テキスト分割を開始します")
        
        if not segments:
            logger.warning("セグメントが見つかりません。全体を1つのチャンクとして処理します")
            return self._create_single_chunk(text)
        
        # セグメントを位置でソート
        sorted_segments = sorted(segments, key=lambda x: x.start_pos)
        
        # チャンクを作成
        chunks = []
        
        for i, segment in enumerate(sorted_segments):
            # 次のセグメントの開始位置を取得
            next_start = sorted_segments[i + 1].start_pos if i + 1 < len(sorted_segments) else len(text)
            
            # コンテンツを抽出（セグメントのタイトル位置から次のセグメントまで）
            content_start = segment.start_pos
            content_end = next_start
            
            # 実際のコンテンツを抽出
            raw_content = text[content_start:content_end].strip()
            
            # コンテンツが空でない場合のみ処理
            if raw_content:
                # AIによる内容の精査（長いコンテンツの場合のみ）
                if len(raw_content) > 200:
                    refined_content = self._refine_content_with_ai(segment, raw_content, raw_content)
                else:
                    refined_content = raw_content
                
                chunk = TextChunk(
                    type=segment.type,
                    number=segment.number,
                    title=segment.title,
                    content=refined_content,
                    start_pos=segment.start_pos,
                    end_pos=content_end,
                    parent_chapter=segment.parent_chapter,
                    word_count=len(refined_content)
                )
                
                chunks.append(chunk)
                logger.debug(f"チャンク作成: {segment.type} {segment.number} - {len(refined_content)}文字")
            else:
                logger.warning(f"空のコンテンツをスキップ: {segment.type} {segment.number}")
        
        # 境界調整を実行
        logger.info("境界調整を開始します")
        adjusted_chunks = self.boundary_adjuster.adjust_boundaries(chunks, text)
        
        self.chunks = adjusted_chunks
        logger.info(f"テキスト分割・境界調整完了: {len(adjusted_chunks)}個のチャンク")
        
        return adjusted_chunks
    
    def _create_single_chunk(self, text: str) -> List[TextChunk]:
        """
        単一チャンクを作成する（構造が検出されない場合）
        
        Args:
            text: 入力テキスト
            
        Returns:
            List[TextChunk]: 単一チャンクリスト
        """
        # AIでタイトルを推定
        estimated_title = self._estimate_title_with_ai(text)
        
        chunk = TextChunk(
            type='chapter',
            number='1',
            title=estimated_title,
            content=clean_text(text),
            start_pos=0,
            end_pos=len(text),
            word_count=len(text)
        )
        
        return [chunk]
    
    def _refine_content_with_ai(self, segment: StructureSegment, full_content: str, body_content: str) -> str:
        """
        AIを使用してコンテンツを精査し、文章の完結性を確保する
        
        Args:
            segment: 構造セグメント
            full_content: タイトル含む全コンテンツ
            body_content: 本文のみのコンテンツ
            
        Returns:
            str: 精査されたコンテンツ
        """
        # コンテンツが短すぎる場合はそのまま返す
        if len(body_content) < 100:
            return body_content
        
        # コンテンツが長すぎる場合は先頭部分のみをAIで分析
        analysis_content = body_content[:2000] if len(body_content) > 2000 else body_content
        
        system_prompt = """あなたは書籍編集の専門家です。
与えられたテキストセクションを分析し、以下の作業を行ってください：

1. 不要な改行や空白を整理
2. 明らかに次のセクションに属する内容があれば除外
3. セクションの境界を適切に調整し、文章が完結するようにする
4. 接続詞や助詞で始まる不完全な文章を修正
5. 内容の一貫性を確保

重要：このセクションが独立した完結した文章になるように調整してください。
整理されたテキストのみを返してください。説明は不要です。"""
        
        user_prompt = f"""以下のセクションを整理し、完結した文章にしてください：

セクション情報:
- タイプ: {segment.type}
- 番号: {segment.number}
- タイトル: {segment.title}

コンテンツ:
{analysis_content}

このセクションが「ために、」「そして、」などの接続詞で始まらず、適切な句読点で終わる完結した文章になるように調整してください。
"""
        
        try:
            response = self.client.ask(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=1500,
                temperature=0.1
            )
            
            refined_content = self.client.extract_text(response).strip()
            
            # 文章完結性をチェック
            if not self._is_content_complete(refined_content):
                logger.warning(f"AI精査結果が不完全です。元のコンテンツを使用: {segment.number}")
                return body_content
            
            # AIの結果が元のコンテンツより大幅に短い場合は元を使用
            if len(refined_content) < len(body_content) * 0.3:
                logger.warning(f"AI精査結果が短すぎます。元のコンテンツを使用: {segment.number}")
                return body_content
            
            # 長いコンテンツの場合は、精査された部分と残りを結合
            if len(body_content) > 2000:
                remaining_content = body_content[2000:]
                # 残りの部分も文章境界で調整
                adjusted_remaining = self._adjust_remaining_content(remaining_content)
                return refined_content + "\n\n" + adjusted_remaining
            
            return refined_content
            
        except Exception as e:
            logger.error(f"AI精査中にエラーが発生しました: {e}")
            return body_content
    
    def _is_content_complete(self, content: str) -> bool:
        """
        コンテンツが完結した文章かどうかをチェック
        
        Args:
            content: チェックするコンテンツ
            
        Returns:
            bool: 完結していればTrue
        """
        import re
        
        content = content.strip()
        if not content:
            return False
        
        # 不完全な開始パターンをチェック
        incomplete_start_patterns = [
            r'^[ため、そして、また、しかし、だが、それで、つまり、すなわち、なお、ただし、もっとも、ところで、さて、では、そこで、こうして、このように、そのため、したがって、ゆえに、よって]',
            r'^[の、が、を、に、へ、と、で、から、より、まで、について、における、に関して、に対して、によって、として、という、といった、などの、などを]',
            r'^[て、で、し、く、き、い、う、る、れ、ろ、ん]',
            r'^[、。！？]'
        ]
        
        for pattern in incomplete_start_patterns:
            if re.match(pattern, content):
                return False
        
        # 適切な終了をチェック
        if not re.search(r'[。！？]', content):
            return False
        
        return True
    
    def _adjust_remaining_content(self, remaining_content: str) -> str:
        """
        残りのコンテンツの境界を調整
        
        Args:
            remaining_content: 残りのコンテンツ
            
        Returns:
            str: 調整されたコンテンツ
        """
        # 文の境界で適切に開始するように調整
        lines = remaining_content.split('\n')
        adjusted_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('ため、', 'そして、', 'また、', 'しかし、', 'だが、')):
                adjusted_lines.append(line)
            elif adjusted_lines:  # 前の行に結合
                adjusted_lines[-1] += line
        
        return '\n'.join(adjusted_lines)
    
    def _estimate_title_with_ai(self, text: str) -> str:
        """
        AIを使用してタイトルを推定する
        
        Args:
            text: 入力テキスト
            
        Returns:
            str: 推定タイトル
        """
        # 先頭部分のみを使用
        analysis_text = text[:1000] if len(text) > 1000 else text
        
        system_prompt = """あなたは書籍分析の専門家です。
与えられたテキストから適切な書籍タイトルを推定してください。
タイトルのみを返してください。説明は不要です。"""
        
        user_prompt = f"""以下のテキストから書籍タイトルを推定してください：

{analysis_text}
"""
        
        try:
            response = self.client.ask(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=100,
                temperature=0.3
            )
            
            title = self.client.extract_text(response).strip()
            return title if title else "タイトル未設定"
            
        except Exception as e:
            logger.error(f"タイトル推定中にエラーが発生しました: {e}")
            return "タイトル未設定"
    
    def optimize_chunks(self) -> List[TextChunk]:
        """
        チャンクを最適化する
        
        Returns:
            List[TextChunk]: 最適化されたチャンクリスト
        """
        if not self.chunks:
            return []
        
        logger.info("チャンクの最適化を開始します")
        
        optimized_chunks = []
        
        for chunk in self.chunks:
            # 空のチャンクをスキップ
            if not chunk.content.strip():
                logger.warning(f"空のチャンクをスキップ: {chunk.type} {chunk.number}")
                continue
            
            # 短すぎるチャンクを前のチャンクと結合
            if len(optimized_chunks) > 0 and len(chunk.content) < 50:
                logger.info(f"短いチャンクを結合: {chunk.type} {chunk.number}")
                prev_chunk = optimized_chunks[-1]
                prev_chunk.content += "\n\n" + chunk.content
                prev_chunk.word_count = len(prev_chunk.content)
                prev_chunk.end_pos = chunk.end_pos
                continue
            
            optimized_chunks.append(chunk)
        
        logger.info(f"チャンク最適化完了: {len(optimized_chunks)}個のチャンク")
        return optimized_chunks
    
    def get_split_summary(self) -> Dict[str, Any]:
        """
        分割結果の要約を取得する
        
        Returns:
            Dict[str, Any]: 分割要約
        """
        if not self.chunks:
            return {'total_chunks': 0, 'total_words': 0}
        
        chapters = [c for c in self.chunks if c.type == 'chapter']
        sections = [c for c in self.chunks if c.type == 'section']
        total_words = sum(c.word_count for c in self.chunks)
        
        return {
            'total_chunks': len(self.chunks),
            'total_chapters': len(chapters),
            'total_sections': len(sections),
            'total_words': total_words,
            'average_chunk_size': total_words // len(self.chunks) if self.chunks else 0,
            'chunk_sizes': [c.word_count for c in self.chunks]
        }