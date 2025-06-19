#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YAML Splitter - YAMLファイル分割ツール
==================================================

このプログラムは、YAML形式のファイルを読み込み、その内容を複数のテキストファイルに分割するツールです。
主に書籍構造化分割ツール（TextBookStructurer）で生成されたYAMLファイルを処理するために設計されています。

主な機能:
-------
- YAML形式のファイルを読み込み
- 章・節ごとに個別のテキストファイルに分割
- ファイル名は「(6桁連番)(タイトル)_(サマリー).txt」の形式
- ファイル内容の先頭にサマリーを追加
- ファイル名の長さ制限による自動切り詰め

使用方法:
-------
```
python yaml_splitter.py 入力ファイル
```

例:
```
python yaml_splitter.py output.yaml
```

入力ファイル: TextBookStructurerで生成されたYAML形式のファイル

出力:
入力ファイルと同じディレクトリに、章・節ごとのテキストファイルが生成されます。
ファイル名の形式: 000001タイトル_サマリー.txt

注意事項:
-------
- 入力ファイルは有効なYAML形式である必要があります
- ファイル名が長すぎる場合は自動的に切り詰められます
"""

import os
import sys
import yaml
import re


def sanitize_filename(filename, max_length=50):
    """
    ファイル名に使用できない文字を置換し、長さを制限する
    
    Args:
        filename: 元のファイル名
        max_length: ファイル名の最大長（デフォルト: 50文字）
        
    Returns:
        str: 安全なファイル名
    """
    # ファイル名に使用できない文字を置換
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    # 長さを制限
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def process_yaml_file(yaml_file_path):
    """
    YAMLファイルを処理して複数のテキストファイルに分割する
    
    Args:
        yaml_file_path: YAMLファイルのパス
    """
    # 出力ディレクトリはinputfileと同じディレクトリ
    output_dir = os.path.dirname(yaml_file_path)
    if not output_dir:
        output_dir = '.'
    
    print(f"YAMLファイルを処理中: {yaml_file_path}")
    print(f"出力ディレクトリ: {output_dir}")
    
    try:
        # YAMLファイルを読み込む
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        if not yaml_data:
            print("エラー: YAMLファイルが空または無効です")
            return
        
        # 書籍タイトルを取得
        book_title = yaml_data.get('book_title', 'タイトル未設定')
        print(f"書籍タイトル: {book_title}")
        
        # ファイル連番の初期値
        file_counter = 1
        
        # 章を処理
        chapters = yaml_data.get('chapters', [])
        for chapter in chapters:
            chapter_number = chapter.get('number', '')
            chapter_title = chapter.get('title', f'章 {chapter_number}')
            chapter_summary = chapter.get('summary', '')
            chapter_content = chapter.get('content', '')
            
            # 章のファイルを作成
            file_counter = create_text_file(
                output_dir,
                file_counter,
                chapter_number,
                chapter_title,
                chapter_summary,
                chapter_content
            )
            
            # 節を処理
            sections = chapter.get('sections', [])
            for section in sections:
                section_number = section.get('number', '')
                section_title = section.get('title', f'節 {section_number}')
                section_summary = section.get('summary', '')
                section_content = section.get('content', '')
                
                # 節のファイルを作成
                file_counter = create_text_file(
                    output_dir,
                    file_counter,
                    section_number,
                    section_title,
                    section_summary,
                    section_content
                )
        
        print(f"処理完了: {file_counter - 1}個のファイルを作成しました")
        
    except Exception as e:
        print(f"エラー: {e}")


def create_text_file(output_dir, file_counter, number, title, summary, content):
    """
    テキストファイルを作成する
    
    Args:
        output_dir: 出力ディレクトリ
        file_counter: ファイル連番
        number: 章または節の番号
        title: タイトル
        summary: 要約
        content: 内容
        
    Returns:
        int: 更新されたファイル連番
    """
    if not content.strip():
        return file_counter
    
    # ファイル名を作成
    padded_counter = str(file_counter).zfill(6)
    
    # タイトルとサマリーの長さを制限
    # カウンター(6桁) + 拡張子(.txt)で10文字使用するため、残りを分配
    title_max_length = 30
    summary_max_length = 30
    
    safe_title = sanitize_filename(title, title_max_length)
    safe_summary = sanitize_filename(summary, summary_max_length)
    
    if safe_summary:
        filename = f"{padded_counter}{safe_title}_{safe_summary}.txt"
    else:
        filename = f"{padded_counter}{safe_title}.txt"
    
    file_path = os.path.join(output_dir, filename)
    
    # ファイルを作成
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # サマリーをファイルの先頭に追加
            if summary:
                f.write(f"summary: {summary}\n\n")
            # 本文を書き込み
            f.write(content)
        print(f"ファイル作成: {filename}")
        return file_counter + 1
    except Exception as e:
        print(f"ファイル作成エラー ({filename}): {e}")
        return file_counter + 1


def main():
    """メイン関数"""
    # コマンドライン引数をチェック
    if len(sys.argv) != 2:
        print("使用方法: python yaml_splitter.py inputfile")
        sys.exit(1)
    
    yaml_file_path = sys.argv[1]
    
    # ファイルの存在をチェック
    if not os.path.exists(yaml_file_path):
        print(f"エラー: ファイルが見つかりません: {yaml_file_path}")
        sys.exit(1)
    
    # YAMLファイルを処理
    process_yaml_file(yaml_file_path)


if __name__ == "__main__":
    main()