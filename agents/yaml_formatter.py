#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YAMLFormatterAgent - 出力整形エージェント

YAML構造に整形し、ファイルへ保存するエージェント
"""

import logging
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path

from utils.helpers import validate_yaml_structure, escape_yaml_string
from agents.labeler import EnrichedChunk

logger = logging.getLogger(__name__)


class YAMLFormatterAgent:
    """
    YAML出力整形エージェント
    
    エンリッチされたチャンクをYAML構造に整形し、ファイルに保存する
    """
    
    def __init__(self):
        """初期化"""
        self.yaml_data = {}
        
    def format_to_yaml(self, chunks: List[EnrichedChunk], book_title: str = "タイトル未設定") -> Dict[str, Any]:
        """
        チャンクをYAML構造に整形する
        
        Args:
            chunks: エンリッチ済みチャンクリスト
            book_title: 書籍タイトル
            
        Returns:
            Dict[str, Any]: YAML構造データ
        """
        logger.info("YAML構造への整形を開始します")
        
        # 基本構造を作成
        yaml_data = {
            'book_title': book_title,
            'chapters': []
        }
        
        # チャンクを章・節に分類
        chapters_dict = {}
        sections_dict = {}
        
        for chunk in chunks:
            if chunk.type == 'chapter':
                chapters_dict[chunk.number] = chunk
            elif chunk.type == 'section':
                if chunk.parent_chapter:
                    if chunk.parent_chapter not in sections_dict:
                        sections_dict[chunk.parent_chapter] = []
                    sections_dict[chunk.parent_chapter].append(chunk)
                else:
                    # 親章が不明な節は独立した章として扱う
                    chapters_dict[chunk.number] = chunk
        
        # 章を番号順にソート
        sorted_chapters = sorted(chapters_dict.items(), key=lambda x: self._extract_number(x[0]))
        
        # YAML構造を構築
        for chapter_num, chapter_chunk in sorted_chapters:
            chapter_data = self._create_chapter_data(chapter_chunk)
            
            # 対応する節を追加
            chapter_number = self._extract_number(chapter_num)
            if chapter_number in sections_dict:
                chapter_sections = sections_dict[chapter_number]
                # 節を番号順にソート
                sorted_sections = sorted(chapter_sections, key=lambda x: x.number)
                
                chapter_data['sections'] = []
                for section in sorted_sections:
                    section_data = self._create_section_data(section)
                    chapter_data['sections'].append(section_data)
            
            yaml_data['chapters'].append(chapter_data)
        
        # 章が存在しない場合の処理
        if not yaml_data['chapters'] and chunks:
            logger.warning("章が検出されませんでした。全コンテンツを単一章として処理します")
            combined_content = "\n\n".join([chunk.content for chunk in chunks])
            combined_summary = " ".join([chunk.summary for chunk in chunks if chunk.summary])
            
            single_chapter = {
                'number': 1,
                'title': chunks[0].title if chunks else "全体",
                'summary': combined_summary[:200] if combined_summary else "",
                'content': combined_content
            }
            yaml_data['chapters'].append(single_chapter)
        
        # 構造の検証
        if validate_yaml_structure(yaml_data):
            logger.info("YAML構造の検証が成功しました")
        else:
            logger.warning("YAML構造の検証で問題が検出されました")
        
        self.yaml_data = yaml_data
        logger.info(f"YAML整形完了: {len(yaml_data['chapters'])}章")
        
        return yaml_data
    
    def _extract_number(self, number_str: str) -> int:
        """
        番号文字列から数値を抽出する
        
        Args:
            number_str: 番号文字列
            
        Returns:
            int: 抽出された数値
        """
        import re
        
        # 数字を抽出
        match = re.search(r'(\d+)', str(number_str))
        if match:
            return int(match.group(1))
        return 0
    
    def _create_chapter_data(self, chunk: EnrichedChunk) -> Dict[str, Any]:
        """
        章データを作成する
        
        Args:
            chunk: エンリッチ済みチャンク
            
        Returns:
            Dict[str, Any]: 章データ
        """
        chapter_data = {
            'number': self._extract_number(chunk.number),
            'title': chunk.title,
            'content': self._format_content(chunk.content)
        }
        
        # オプショナルフィールドを追加
        if chunk.summary:
            chapter_data['summary'] = chunk.summary
        
        if chunk.keywords:
            chapter_data['keywords'] = chunk.keywords
        
        # estimated_reading_timeの追加を削除
        
        return chapter_data
    
    def _create_section_data(self, chunk: EnrichedChunk) -> Dict[str, Any]:
        """
        節データを作成する
        
        Args:
            chunk: エンリッチ済みチャンク
            
        Returns:
            Dict[str, Any]: 節データ
        """
        section_data = {
            'number': chunk.number,
            'title': chunk.title,
            'content': self._format_content(chunk.content)
        }
        
        # オプショナルフィールドを追加
        if chunk.summary:
            section_data['summary'] = chunk.summary
        
        if chunk.keywords:
            section_data['keywords'] = chunk.keywords
        
        return section_data
    
    def _format_content(self, content: str) -> str:
        """
        コンテンツをYAML出力用にフォーマットする
        
        Args:
            content: 元のコンテンツ
            
        Returns:
            str: フォーマット済みコンテンツ
        """
        if not content:
            return ""
        
        # 改行の正規化
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # 余分な空行を削除
        lines = content.split('\n')
        formatted_lines = []
        prev_empty = False
        
        for line in lines:
            line = line.rstrip()
            if not line:
                if not prev_empty:
                    formatted_lines.append('')
                prev_empty = True
            else:
                formatted_lines.append(line)
                prev_empty = False
        
        # 末尾の空行を削除
        while formatted_lines and not formatted_lines[-1]:
            formatted_lines.pop()
        
        return '\n'.join(formatted_lines)
    
    def save_to_file(self, output_path: str, yaml_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        YAML データをファイルに保存する
        
        Args:
            output_path: 出力ファイルパス
            yaml_data: 保存するYAMLデータ（Noneの場合は内部データを使用）
            
        Returns:
            bool: 保存成功フラグ
        """
        data_to_save = yaml_data or self.yaml_data
        
        if not data_to_save:
            logger.error("保存するYAMLデータがありません")
            return False
        
        try:
            # 出力ディレクトリを作成
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # YAMLファイルとして保存
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(
                    data_to_save,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                    width=1000,  # 長い行の折り返しを防ぐ
                    sort_keys=False  # キーの順序を保持
                )
            
            logger.info(f"YAMLファイルを保存しました: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"YAMLファイルの保存中にエラーが発生しました: {e}")
            return False
    
    def generate_yaml_string(self, yaml_data: Optional[Dict[str, Any]] = None) -> str:
        """
        YAML文字列を生成する
        
        Args:
            yaml_data: YAML化するデータ（Noneの場合は内部データを使用）
            
        Returns:
            str: YAML文字列
        """
        data_to_convert = yaml_data or self.yaml_data
        
        if not data_to_convert:
            return ""
        
        try:
            yaml_string = yaml.dump(
                data_to_convert,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
                width=1000,
                sort_keys=False
            )
            
            return yaml_string
            
        except Exception as e:
            logger.error(f"YAML文字列の生成中にエラーが発生しました: {e}")
            return ""
    
    def add_metadata(self, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        YAMLデータにメタデータを追加する
        
        Args:
            yaml_data: 元のYAMLデータ
            
        Returns:
            Dict[str, Any]: メタデータ追加済みYAMLデータ
        """
        from datetime import datetime
        
        # メタデータを追加
        enhanced_data = yaml_data.copy()
        
        # 生成情報を追加
        enhanced_data['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'generator': 'TextBookStructurer',
            'version': '1.0.0',
            'total_chapters': len(yaml_data.get('chapters', [])),
            'total_sections': sum(
                len(chapter.get('sections', [])) 
                for chapter in yaml_data.get('chapters', [])
            )
        }
        
        # 統計情報を追加
        total_content_length = 0
        total_reading_time = 0
        
        for chapter in yaml_data.get('chapters', []):
            if 'content' in chapter:
                total_content_length += len(chapter['content'])
            # estimated_reading_timeの集計を削除
            
            for section in chapter.get('sections', []):
                if 'content' in section:
                    total_content_length += len(section['content'])
        
        enhanced_data['metadata']['statistics'] = {
            'total_content_length': total_content_length
            # estimated_total_reading_timeの追加を削除
        }
        
        return enhanced_data
    
    def get_format_summary(self) -> Dict[str, Any]:
        """
        フォーマット結果の要約を取得する
        
        Returns:
            Dict[str, Any]: フォーマット要約
        """
        if not self.yaml_data:
            return {'status': 'no_data'}
        
        chapters = self.yaml_data.get('chapters', [])
        total_sections = sum(len(chapter.get('sections', [])) for chapter in chapters)
        
        return {
            'status': 'completed',
            'book_title': self.yaml_data.get('book_title', ''),
            'total_chapters': len(chapters),
            'total_sections': total_sections,
            'has_summaries': any(
                'summary' in chapter for chapter in chapters
            ),
            'has_keywords': any(
                'keywords' in chapter for chapter in chapters
            )
        }