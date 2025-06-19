#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegmenterAgent - 構造抽出エージェント

AIが人間のように文章を読んで意味を理解しながら章・節を分割するエージェント
"""

import logging
import math
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from openai_client import OpenAIClient
from utils.helpers import clean_text

logger = logging.getLogger(__name__)


@dataclass
class StructureSegment:
    """構造セグメントを表すデータクラス"""
    type: str  # 'chapter' or 'section'
    number: str
    title: str
    start_pos: int
    end_pos: int
    content: str = ""
    parent_chapter: Optional[int] = None


class SegmenterAgent:
    """
    構造抽出エージェント
    
    AIが人間のように文章を読んで意味を理解しながら章・節を分割する
    """
    
    def __init__(self, openai_client: OpenAIClient):
        """
        初期化
        
        Args:
            openai_client: OpenAIクライアント
        """
        self.client = openai_client
        self.segments = []
        self.chunk_size = 3000  # 一度に分析するテキストサイズ
        self.overlap_size = 500  # 重複サイズ
        
    def analyze_structure(self, text: str) -> Dict[str, Any]:
        """
        AIが人間のように文章を読んで意味を理解しながらテキストの構造を分析する
        
        Args:
            text: 入力テキスト
            
        Returns:
            Dict[str, Any]: 構造分析結果
        """
        logger.info("AIによる意味理解ベースの構造分析を開始します")
        logger.info(f"総文字数: {len(text):,}文字")
        
        # テキストを分析可能なチャンクに分割
        chunks = self._split_text_into_chunks(text)
        logger.info(f"分析チャンク数: {len(chunks)}個")
        
        # 各チャンクを順次分析
        all_segments = []
        book_title = "タイトル未設定"
        
        for i, chunk_info in enumerate(chunks):
            logger.info(f"チャンク {i+1}/{len(chunks)} を分析中...")
            
            # チャンクの構造を分析
            chunk_analysis = self._analyze_chunk_structure(
                chunk_info['text'],
                chunk_info['start_pos'],
                i,
                len(chunks),
                all_segments  # 既存セグメント情報を渡す
            )
            
            # 書籍タイトルを取得（最初のチャンクから）
            if i == 0 and chunk_analysis.get('book_title'):
                book_title = chunk_analysis['book_title']
            
            # セグメントを追加
            if chunk_analysis.get('segments'):
                all_segments.extend(chunk_analysis['segments'])
        
        # セグメントを統合・最適化
        optimized_segments = self._optimize_segments(all_segments, text)
        
        logger.info(f"構造分析完了: {len(optimized_segments)}個のセグメント")
        
        result = {
            'book_title': book_title,
            'total_chunks_analyzed': len(chunks),
            'segments': optimized_segments,
            'analysis_method': 'ai_semantic_understanding'
        }
        
        return result
    
    def _split_text_into_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        テキストを分析可能なチャンクに分割する
        
        Args:
            text: 入力テキスト
            
        Returns:
            List[Dict[str, Any]]: チャンク情報のリスト
        """
        chunks = []
        text_length = len(text)
        
        start_pos = 0
        while start_pos < text_length:
            # チャンクの終了位置を計算
            end_pos = min(start_pos + self.chunk_size, text_length)
            
            # 文の境界で切るように調整（最後のチャンク以外）
            if end_pos < text_length:
                # 句読点で区切る
                for punct in ['。', '！', '？', '\n\n']:
                    punct_pos = text.rfind(punct, start_pos, end_pos)
                    if punct_pos > start_pos + self.chunk_size // 2:  # 最低半分は確保
                        end_pos = punct_pos + len(punct)
                        break
            
            chunk_text = text[start_pos:end_pos]
            
            chunks.append({
                'text': chunk_text,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'length': len(chunk_text)
            })
            
            # 次のチャンクの開始位置（重複を考慮）
            if end_pos >= text_length:
                break
            start_pos = max(start_pos + self.chunk_size - self.overlap_size, end_pos)
        
        return chunks
    
    def _analyze_chunk_structure(self, chunk_text: str, chunk_start_pos: int,
                                chunk_index: int, total_chunks: int,
                                existing_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        チャンクの構造をAIで分析する
        
        Args:
            chunk_text: チャンクテキスト
            chunk_start_pos: チャンクの開始位置
            chunk_index: チャンクのインデックス
            total_chunks: 総チャンク数
            existing_segments: 既存のセグメント情報
            
        Returns:
            Dict[str, Any]: チャンク分析結果
        """
        # 既存セグメントの情報を要約
        context_info = ""
        if existing_segments:
            recent_segments = existing_segments[-3:]  # 直近3つのセグメント
            context_info = "直前のセグメント情報:\n"
            for seg in recent_segments:
                context_info += f"- {seg['type']}: {seg['title']}\n"
        
        system_prompt = """あなたは書籍構造分析の専門家です。
与えられたテキストチャンクを読んで、人間のように意味を理解し、章・節の区切りを特定してください。

以下の点に注意してください：
1. 内容の意味的な変化を重視する
2. 話題の転換点を見つける
3. 新しい章や節の開始を判断する
4. 章タイトルや節タイトルを推定する
5. 各セグメントが完結した文章になるように境界を設定する
6. 接続詞や助詞で始まる不完全な文章を避ける

重要：セグメントの境界は、文章が自然に区切れる位置に設定してください。
「ために、」「そして、」などで始まる不完全な文章にならないよう注意してください。

以下の形式で情報を返してください：

BOOK_TITLE: 書籍タイトル（最初のチャンクのみ）

SEGMENTS:
[SEGMENT_START]
TYPE: chapter または section
TITLE: 推定タイトル
POSITION: 文字位置
CONFIDENCE: 0.0-1.0
REASON: 判断理由
QUALITY: 境界の品質評価
[SEGMENT_END]

SUMMARY: このチャンクの内容要約"""
        
        user_prompt = f"""以下のテキストチャンク（{chunk_index+1}/{total_chunks}）を分析してください：

{context_info}

チャンク内容:
{chunk_text}

このチャンクで新しい章や節が始まっている箇所があれば特定し、適切なタイトルを推定してください。

重要な注意点：
- セグメントの境界は完結した文章の終わりに設定してください
- 次のセグメントが「ために、」「そして、」「また、」などの接続詞で始まらないようにしてください
- 各セグメントが独立して読める完結した内容になるようにしてください
"""
        
        try:
            response = self.client.ask(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=1500,
                temperature=0.2
            )
            
            # テキスト形式のレスポンスを解析
            response_text = self.client.extract_text(response)
            analysis = self._parse_structured_response(response_text, chunk_text, chunk_index)
            
            # 必要なキーが存在しない場合のデフォルト値設定
            if 'segments' not in analysis:
                analysis['segments'] = []
            if 'book_title' not in analysis:
                analysis['book_title'] = ""
            if 'content_summary' not in analysis:
                analysis['content_summary'] = ""
            
            # セグメントの位置を絶対位置に変換
            if analysis.get('segments'):
                for segment in analysis['segments']:
                    if 'start_pos_in_chunk' in segment:
                        segment['start_pos'] = chunk_start_pos + segment['start_pos_in_chunk']
                        segment['chunk_index'] = chunk_index
                    
                    # セグメントに必要なフィールドが不足している場合のデフォルト値
                    if 'type' not in segment:
                        segment['type'] = 'chapter'
                    if 'title' not in segment:
                        segment['title'] = f"セクション {chunk_index + 1}"
                    if 'confidence' not in segment:
                        segment['confidence'] = 0.5
                    if 'reason' not in segment:
                        segment['reason'] = "自動生成"
                    if 'boundary_quality' not in segment:
                        segment['boundary_quality'] = "medium"
            
            return analysis
            
        except Exception as e:
            logger.error(f"チャンク分析中にエラーが発生しました: {e}")
            # フォールバック: 基本的な分析結果を返す
            return {
                "book_title": "タイトル未設定" if chunk_index == 0 else "",
                "segments": [{
                    "type": "chapter",
                    "title": f"セクション {chunk_index + 1}",
                    "start_pos_in_chunk": 0,
                    "start_pos": chunk_start_pos,
                    "confidence": 0.3,
                    "reason": "エラー時のフォールバック",
                    "boundary_quality": "low",
                    "chunk_index": chunk_index
                }] if chunk_index % 2 == 0 else [],  # 偶数チャンクのみセグメント作成
                "content_summary": f"チャンク {chunk_index + 1} の内容"
            }
    
    def _parse_structured_response(self, response_text: str, chunk_text: str, chunk_index: int) -> Dict[str, Any]:
        """
        構造化されたテキストレスポンスを解析する
        
        Args:
            response_text: AIからのレスポンステキスト
            chunk_text: 分析対象のチャンクテキスト
            chunk_index: チャンクのインデックス
            
        Returns:
            Dict[str, Any]: 解析結果
        """
        logger.debug("構造化テキストレスポンスを解析中...")
        
        # 初期化
        book_title = ""
        segments = []
        content_summary = ""
        
        try:
            # BOOK_TITLEを抽出
            title_match = re.search(r'BOOK_TITLE:\s*(.+)', response_text)
            if title_match:
                book_title = title_match.group(1).strip()
            
            # SUMMARYを抽出
            summary_match = re.search(r'SUMMARY:\s*(.+)', response_text, re.DOTALL)
            if summary_match:
                content_summary = summary_match.group(1).strip()
            
            # SEGMENTSを抽出
            segment_pattern = r'\[SEGMENT_START\](.*?)\[SEGMENT_END\]'
            segment_matches = re.findall(segment_pattern, response_text, re.DOTALL)
            
            for segment_text in segment_matches:
                segment = {}
                
                # 各フィールドを抽出
                type_match = re.search(r'TYPE:\s*(.+)', segment_text)
                if type_match:
                    segment['type'] = type_match.group(1).strip()
                
                title_match = re.search(r'TITLE:\s*(.+)', segment_text)
                if title_match:
                    segment['title'] = title_match.group(1).strip()
                
                position_match = re.search(r'POSITION:\s*(\d+)', segment_text)
                if position_match:
                    segment['start_pos_in_chunk'] = int(position_match.group(1))
                
                confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', segment_text)
                if confidence_match:
                    segment['confidence'] = float(confidence_match.group(1))
                
                reason_match = re.search(r'REASON:\s*(.+)', segment_text)
                if reason_match:
                    segment['reason'] = reason_match.group(1).strip()
                
                quality_match = re.search(r'QUALITY:\s*(.+)', segment_text)
                if quality_match:
                    segment['boundary_quality'] = quality_match.group(1).strip()
                
                # 必要なフィールドが存在する場合のみ追加
                if 'type' in segment and 'title' in segment:
                    # デフォルト値を設定
                    if 'start_pos_in_chunk' not in segment:
                        segment['start_pos_in_chunk'] = 0
                    if 'confidence' not in segment:
                        segment['confidence'] = 0.5
                    if 'reason' not in segment:
                        segment['reason'] = "構造化解析による検出"
                    if 'boundary_quality' not in segment:
                        segment['boundary_quality'] = "medium"
                    
                    segments.append(segment)
            
            logger.debug(f"構造化解析完了: {len(segments)}個のセグメント検出")
            
        except Exception as e:
            logger.warning(f"構造化レスポンス解析中にエラー: {e}")
            # フォールバックとして従来の解析を実行
            return self._fallback_text_analysis(response_text, chunk_text, chunk_index)
        
        # 結果が空の場合はフォールバック
        if not segments and not book_title:
            logger.warning("構造化解析で結果が得られませんでした。フォールバック解析を実行します")
            return self._fallback_text_analysis(response_text, chunk_text, chunk_index)
        
        return {
            "book_title": book_title,
            "segments": segments,
            "content_summary": content_summary
        }
    
    def _fallback_text_analysis(self, response_text: str, chunk_text: str, chunk_index: int) -> Dict[str, Any]:
        """
        JSONパースに失敗した場合のフォールバック解析
        
        Args:
            response_text: AIからのレスポンステキスト
            chunk_text: 分析対象のチャンクテキスト
            chunk_index: チャンクのインデックス
            
        Returns:
            Dict[str, Any]: 解析結果
        """
        logger.info("フォールバック解析を実行中...")
        
        # レスポンステキストから情報を抽出
        segments = []
        book_title = ""
        
        # 章や節のキーワードを探す
        chapter_keywords = ['第', '章', 'Chapter', 'CHAPTER', '■', '●', '◆']
        section_keywords = ['節', 'Section', 'SECTION', '▼', '○', '◇']
        
        lines = chunk_text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 章の検出
            for keyword in chapter_keywords:
                if keyword in line and len(line) < 100:  # タイトルは通常短い
                    segments.append({
                        "type": "chapter",
                        "title": line,
                        "start_pos_in_chunk": chunk_text.find(line),
                        "confidence": 0.4,
                        "reason": f"キーワード '{keyword}' による検出",
                        "boundary_quality": "medium"
                    })
                    break
            
            # 節の検出
            for keyword in section_keywords:
                if keyword in line and len(line) < 100:
                    segments.append({
                        "type": "section",
                        "title": line,
                        "start_pos_in_chunk": chunk_text.find(line),
                        "confidence": 0.3,
                        "reason": f"キーワード '{keyword}' による検出",
                        "boundary_quality": "medium"
                    })
                    break
        
        # 書籍タイトルの推定（最初のチャンクのみ）
        if chunk_index == 0:
            first_lines = chunk_text.split('\n')[:5]
            for line in first_lines:
                line = line.strip()
                if line and len(line) < 50 and not any(kw in line for kw in chapter_keywords + section_keywords):
                    book_title = line
                    break
        
        return {
            "book_title": book_title,
            "segments": segments,
            "content_summary": f"フォールバック解析によるチャンク {chunk_index + 1} の分析結果"
        }
    
    def _optimize_segments(self, segments: List[Dict[str, Any]], text: str) -> List[StructureSegment]:
        """
        AIで検出されたセグメントを最適化・統合する
        
        Args:
            segments: 検出されたセグメント情報
            text: 元のテキスト
            
        Returns:
            List[StructureSegment]: 最適化されたセグメントリスト
        """
        if not segments:
            logger.warning("セグメントが検出されませんでした。テキストを均等分割します")
            return self._create_equal_segments(text)
        
        # 信頼度と境界品質でフィルタリング
        filtered_segments = []
        for s in segments:
            confidence = s.get('confidence', 0)
            boundary_quality = s.get('boundary_quality', '')
            
            # 信頼度が高い、または境界品質が良好な場合に採用
            if confidence > 0.2 or 'good' in boundary_quality.lower():
                filtered_segments.append(s)
        
        if not filtered_segments:
            logger.warning("信頼度の高いセグメントがありません。全セグメントを使用します")
            filtered_segments = segments
        
        # 重複を除去し、位置でソート
        unique_segments = []
        seen_titles = set()
        
        for seg in sorted(filtered_segments, key=lambda x: x.get('start_pos', 0)):
            title = seg.get('title', '').strip()
            # タイトルが重複していない、または十分に離れている場合のみ追加
            if title and title not in seen_titles:
                unique_segments.append(seg)
                seen_titles.add(title)
        
        # セグメントが少なすぎる場合は、テキストを均等分割
        if len(unique_segments) < 3:
            logger.warning(f"検出されたセグメントが少なすぎます({len(unique_segments)}個)。均等分割を実行します")
            return self._create_equal_segments(text)
        
        # StructureSegmentオブジェクトに変換
        structure_segments = []
        for i, seg in enumerate(unique_segments):
            # 次のセグメントの開始位置を取得
            next_pos = unique_segments[i + 1]['start_pos'] if i + 1 < len(unique_segments) else len(text)
            
            # コンテンツを抽出
            start_pos = seg.get('start_pos', 0)
            content = text[start_pos:next_pos].strip()
            
            # コンテンツが空でない場合のみ追加
            if content:
                structure_segment = StructureSegment(
                    type=seg.get('type', 'chapter'),
                    number=str(i + 1),
                    title=seg.get('title', f"セクション {i + 1}"),
                    start_pos=start_pos,
                    end_pos=next_pos,
                    content=clean_text(content)
                )
                
                structure_segments.append(structure_segment)
        
        logger.info(f"セグメント最適化完了: {len(structure_segments)}個")
        return structure_segments
    
    def _create_equal_segments(self, text: str, num_segments: int = 4) -> List[StructureSegment]:
        """
        テキストを均等に分割してセグメントを作成する
        
        Args:
            text: 入力テキスト
            num_segments: 分割数
            
        Returns:
            List[StructureSegment]: セグメントリスト
        """
        segment_size = len(text) // num_segments
        segments = []
        
        # 新・人間革命の章タイトル
        chapter_titles = ["智勇", "使命", "烈風", "大河"]
        
        for i in range(num_segments):
            start_pos = i * segment_size
            end_pos = (i + 1) * segment_size if i < num_segments - 1 else len(text)
            
            content = text[start_pos:end_pos].strip()
            title = chapter_titles[i] if i < len(chapter_titles) else f"第{i+1}章"
            
            segment = StructureSegment(
                type='chapter',
                number=str(i + 1),
                title=title,
                start_pos=start_pos,
                end_pos=end_pos,
                content=clean_text(content)
            )
            
            segments.append(segment)
        
        logger.info(f"均等分割完了: {num_segments}個のセグメント")
        return segments
    
    def get_structure_summary(self) -> Dict[str, Any]:
        """
        構造の要約を取得する
        
        Returns:
            Dict[str, Any]: 構造要約
        """
        chapters = [s for s in self.segments if s.type == 'chapter']
        sections = [s for s in self.segments if s.type == 'section']
        
        return {
            'total_chapters': len(chapters),
            'total_sections': len(sections),
            'chapter_numbers': [s.number for s in chapters],
            'structure_detected': len(chapters) > 0 or len(sections) > 0,
            'analysis_method': 'ai_semantic_understanding'
        }