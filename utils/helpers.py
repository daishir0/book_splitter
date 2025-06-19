#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共通ヘルパー関数群
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    テキストをクリーンアップする
    
    Args:
        text: 入力テキスト
        
    Returns:
        str: クリーンアップされたテキスト
    """
    if not text:
        return ""
    
    # 余分な空白を削除
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 改行の正規化
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text


def extract_chapter_patterns(text: str) -> List[Dict[str, Any]]:
    """
    テキストから章のパターンを抽出する
    
    Args:
        text: 入力テキスト
        
    Returns:
        List[Dict[str, Any]]: 章のパターンリスト
    """
    patterns = []
    
    # 章のパターンを定義（日本語書籍に特化）
    chapter_patterns = [
        # 新・人間革命の章タイトルパターン（智勇、使命、烈風、大河など）
        r'^([智使烈大]\s*[勇命風河])\s*$',  # 智勇、使命、烈風、大河
        r'^([一二三四五六七八九十]+)\s*$',  # 漢数字のみの章
        r'^第([一二三四五六七八九十]+)章\s*(.*)$',  # 第一章 タイトル
        r'^第(\d+)章\s*(.*)$',  # 第1章 タイトル
        r'^第([一二三四五六七八九十]+)編\s*(.*)$',  # 第一編 タイトル
        r'^第(\d+)編\s*(.*)$',  # 第1編 タイトル
        r'^([一二三四五六七八九十]+)、\s*(.+)$',  # 一、タイトル
        r'^(\d+)、\s*(.+)$',  # 1、タイトル
        r'^(\d+)\.\s*(.+)$',  # 1. タイトル
        r'^(\d+)章\s*(.*)$',  # 1章 タイトル
        # 目次から章タイトルを抽出
        r'^《(.+?)》$',  # 《智勇》形式
    ]
    
    for i, pattern in enumerate(chapter_patterns):
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            chapter_num = match.group(1)
            title = clean_text(match.group(2)) if len(match.groups()) > 1 else f"章 {chapter_num}"
            
            patterns.append({
                'type': 'chapter',
                'number': int(chapter_num),
                'title': title,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'pattern_index': i,
                'full_match': match.group(0)
            })
    
    # 位置でソート
    patterns.sort(key=lambda x: x['start_pos'])
    
    return patterns


def extract_section_patterns(text: str) -> List[Dict[str, Any]]:
    """
    テキストから節のパターンを抽出する
    
    Args:
        text: 入力テキスト
        
    Returns:
        List[Dict[str, Any]]: 節のパターンリスト
    """
    patterns = []
    
    # 節のパターンを定義（日本語書籍に特化、ISBN等の誤検出を防ぐ）
    section_patterns = [
        r'^(\d+)\.(\d+)\s+(.+)$',  # 1.1 タイトル（行頭から）
        r'^(\d+)-(\d+)\s+(.+)$',  # 1-1 タイトル（行頭から）
        r'^第(\d+)節\s*(.+)$',  # 第1節 タイトル
        r'^([一二三四五六七八九十]+)節\s*(.+)$',  # 一節 タイトル
        r'^Section\s+(\d+)\.(\d+)\s*[:\-]?\s*(.+)$',  # Section 1.1: Title
    ]
    
    for i, pattern in enumerate(section_patterns):
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            # ISBN番号や電話番号などの誤検出を除外
            full_match = match.group(0)
            if any(exclude in full_match.lower() for exclude in ['isbn', '978-', '979-', 'tel:', 'fax:']):
                continue
            
            if len(match.groups()) >= 3:
                chapter_num = match.group(1)
                section_num = match.group(2)
                title = clean_text(match.group(3))
                section_id = f"{chapter_num}.{section_num}"
            else:
                section_num = match.group(1)
                title = clean_text(match.group(2)) if len(match.groups()) > 1 else f"節 {section_num}"
                section_id = section_num
            
            # タイトルが短すぎる場合はスキップ
            if len(title.strip()) < 2:
                continue
            
            patterns.append({
                'type': 'section',
                'section_id': section_id,
                'title': title,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'pattern_index': i,
                'full_match': match.group(0)
            })
    
    # 位置でソート
    patterns.sort(key=lambda x: x['start_pos'])
    
    return patterns


def split_text_by_positions(text: str, positions: List[int]) -> List[str]:
    """
    指定された位置でテキストを分割する
    
    Args:
        text: 入力テキスト
        positions: 分割位置のリスト
        
    Returns:
        List[str]: 分割されたテキストのリスト
    """
    if not positions:
        return [text]
    
    # 位置をソートして重複を除去
    positions = sorted(set([0] + positions + [len(text)]))
    
    segments = []
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)
    
    return segments


def generate_summary(text: str, max_length: int = 200) -> str:
    """
    テキストの簡単な要約を生成する（AI使用前の仮実装）
    
    Args:
        text: 入力テキスト
        max_length: 最大文字数
        
    Returns:
        str: 要約テキスト
    """
    if not text:
        return ""
    
    # 最初の段落または指定文字数を取得
    paragraphs = text.split('\n\n')
    if paragraphs:
        first_paragraph = paragraphs[0].strip()
        if len(first_paragraph) <= max_length:
            return first_paragraph
        else:
            return first_paragraph[:max_length] + "..."
    
    return text[:max_length] + "..." if len(text) > max_length else text


def validate_yaml_structure(data: Dict[str, Any]) -> bool:
    """
    YAML構造の妥当性を検証する
    
    Args:
        data: 検証するデータ
        
    Returns:
        bool: 妥当性
    """
    try:
        # 必須フィールドの確認
        if 'book_title' not in data:
            logger.warning("book_titleが見つかりません")
            return False
        
        if 'chapters' not in data or not isinstance(data['chapters'], list):
            logger.warning("chaptersが見つからないか、リスト形式ではありません")
            return False
        
        # 各章の構造を確認
        for i, chapter in enumerate(data['chapters']):
            if not isinstance(chapter, dict):
                logger.warning(f"章 {i+1} が辞書形式ではありません")
                return False
            
            required_fields = ['number', 'title']
            for field in required_fields:
                if field not in chapter:
                    logger.warning(f"章 {i+1} に必須フィールド '{field}' が見つかりません")
                    return False
            
            # 節の構造を確認（存在する場合）
            if 'sections' in chapter:
                if not isinstance(chapter['sections'], list):
                    logger.warning(f"章 {i+1} のsectionsがリスト形式ではありません")
                    return False
                
                for j, section in enumerate(chapter['sections']):
                    if not isinstance(section, dict):
                        logger.warning(f"章 {i+1} の節 {j+1} が辞書形式ではありません")
                        return False
                    
                    if 'number' not in section or 'title' not in section:
                        logger.warning(f"章 {i+1} の節 {j+1} に必須フィールドが見つかりません")
                        return False
        
        return True
        
    except Exception as e:
        logger.error(f"YAML構造の検証中にエラーが発生しました: {e}")
        return False


def escape_yaml_string(text: str) -> str:
    """
    YAML文字列をエスケープする
    
    Args:
        text: 入力テキスト
        
    Returns:
        str: エスケープされたテキスト
    """
    if not text:
        return ""
    
    # 特殊文字のエスケープ
    text = text.replace('\\', '\\\\')
    text = text.replace('"', '\\"')
    text = text.replace('\n', '\\n')
    text = text.replace('\r', '\\r')
    text = text.replace('\t', '\\t')
    
    return text