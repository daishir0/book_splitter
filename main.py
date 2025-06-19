#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
書籍構造化分割ツール（TextBookStructurer）
==================================================

このプログラムは、テキスト形式の書籍を解析し、章・節に分割して構造化するツールです。
AIを活用して、テキストの意味を理解しながら適切な分割を行い、YAML形式で出力します。

主な機能:
-------
- AIによるテキスト構造の分析
- 章・節の自動検出と分割
- タイトル・要約・キーワードの自動生成
- 文章の境界調整による完結性の確保
- YAML形式での構造化データ出力

使用方法:
-------
```
python main.py 入力ファイル 出力ファイル
```

例:
```
python main.py input.txt output.yaml
```

入力ファイル: 分析対象のテキストファイル
出力ファイル: 構造化されたYAML形式のファイル

出力されたYAMLファイルは、以下のようなツールで活用できます:
- yaml_splitter.py: YAMLファイルを個別のテキストファイルに分割
- その他のYAML処理ツール

注意事項:
-------
- 大きなテキストファイルの処理には時間がかかる場合があります
- OpenAI APIキーが必要です（config.pyで設定）
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

# LangGraphのインポート
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# 自作モジュールのインポート
from config import DEBUG
from openai_client import OpenAIClient
from agents.segmenter import SegmenterAgent
from agents.splitter import SplitterAgent
from agents.labeler import LabelerAgent
from agents.yaml_formatter import YAMLFormatterAgent

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessingState(TypedDict):
    """処理状態を管理するクラス"""
    input_text: str
    book_title: str
    structure_analysis: Dict[str, Any]
    text_chunks: List[Any]
    enriched_chunks: List[Any]
    yaml_data: Dict[str, Any]
    current_step: str
    error_message: str
    processing_complete: bool


class TextBookStructurer:
    """
    書籍構造化分割ツールのメインクラス
    
    LangGraphを使用してエージェント間の協調処理を管理
    """
    
    def __init__(self, input_file: str, output_file: str):
        """
        初期化
        
        Args:
            input_file: 入力ファイルのパス
            output_file: 出力ファイルのパス
        """
        self.input_file = input_file
        self.output_file = output_file
        
        print("🤖 AIエージェントシステムを初期化中...")
        print("📡 OpenAI APIクライアントを初期化...")
        self.openai_client = OpenAIClient()
        
        print("🔍 SegmenterAgent（構造抽出エージェント）を初期化...")
        self.segmenter = SegmenterAgent(self.openai_client)
        
        print("✂️  SplitterAgent（分割エージェント）を初期化...")
        self.splitter = SplitterAgent(self.openai_client)
        
        print("🏷️  LabelerAgent（メタ付与エージェント）を初期化...")
        self.labeler = LabelerAgent(self.openai_client)
        
        print("📝 YAMLFormatterAgent（出力整形エージェント）を初期化...")
        self.formatter = YAMLFormatterAgent()
        
        print("🔗 LangGraphワークフローを構築中...")
        # LangGraphワークフローを構築
        self.workflow = self._build_workflow()
        print("✅ 全エージェントの初期化が完了しました！")
        print()
        
    def _build_workflow(self) -> StateGraph:
        """
        LangGraphワークフローを構築する
        
        Returns:
            StateGraph: 構築されたワークフロー
        """
        # ワークフローグラフを作成
        workflow = StateGraph(ProcessingState)
        
        # ノード（処理ステップ）を追加
        workflow.add_node("load_input", self._load_input)
        workflow.add_node("analyze_structure", self._analyze_structure)
        workflow.add_node("split_text", self._split_text)
        workflow.add_node("enrich_metadata", self._enrich_metadata)
        workflow.add_node("format_yaml", self._format_yaml)
        workflow.add_node("save_output", self._save_output)
        
        # エッジ（処理フロー）を定義
        workflow.set_entry_point("load_input")
        workflow.add_edge("load_input", "analyze_structure")
        workflow.add_edge("analyze_structure", "split_text")
        workflow.add_edge("split_text", "enrich_metadata")
        workflow.add_edge("enrich_metadata", "format_yaml")
        workflow.add_edge("format_yaml", "save_output")
        workflow.add_edge("save_output", END)
        
        return workflow.compile()
    
    def _load_input(self, state: ProcessingState) -> ProcessingState:
        """
        入力ファイルを読み込む
        
        Args:
            state: 処理状態
            
        Returns:
            ProcessingState: 更新された処理状態
        """
        logger.info("=== ステップ1: 入力ファイル読み込み ===")
        
        try:
            input_path = Path(self.input_file)
            
            if not input_path.exists():
                raise FileNotFoundError(f"入力ファイルが見つかりません: {self.input_file}")
            
            with open(input_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
            
            if not input_text.strip():
                raise ValueError("入力ファイルが空です")
            
            logger.info(f"入力ファイル {self.input_file} 読み込み完了: {len(input_text)}文字")
            
            state["input_text"] = input_text
            state["current_step"] = "load_input_complete"
            state["error_message"] = ""
            
        except Exception as e:
            logger.error(f"入力ファイル読み込みエラー: {e}")
            state["error_message"] = str(e)
            state["current_step"] = "load_input_error"
        
        return state
    
    def _analyze_structure(self, state: ProcessingState) -> ProcessingState:
        """
        テキスト構造を分析する
        
        Args:
            state: 処理状態
            
        Returns:
            ProcessingState: 更新された処理状態
        """
        logger.info("=== ステップ2: 構造分析 ===")
        
        try:
            if state.get("error_message"):
                return state
            
            # SegmenterAgentで構造分析
            structure_analysis = self.segmenter.analyze_structure(state["input_text"])
            
            # 書籍タイトルを取得
            book_title = structure_analysis.get("book_title", "タイトル未設定")
            
            logger.info(f"構造分析完了: {len(structure_analysis['segments'])}個のセグメント")
            
            state["structure_analysis"] = structure_analysis
            state["book_title"] = book_title
            state["current_step"] = "analyze_structure_complete"
            
        except Exception as e:
            logger.error(f"構造分析エラー: {e}")
            state["error_message"] = str(e)
            state["current_step"] = "analyze_structure_error"
        
        return state
    
    def _split_text(self, state: ProcessingState) -> ProcessingState:
        """
        テキストを分割する
        
        Args:
            state: 処理状態
            
        Returns:
            ProcessingState: 更新された処理状態
        """
        logger.info("=== ステップ3: テキスト分割 ===")
        
        try:
            if state.get("error_message"):
                return state
            
            # SplitterAgentでテキスト分割
            segments = state["structure_analysis"]["segments"]
            text_chunks = self.splitter.split_text(state["input_text"], segments)
            
            # チャンクを最適化
            optimized_chunks = self.splitter.optimize_chunks()
            
            logger.info(f"テキスト分割完了: {len(optimized_chunks)}個のチャンク")
            
            state["text_chunks"] = optimized_chunks
            state["current_step"] = "split_text_complete"
            
        except Exception as e:
            logger.error(f"テキスト分割エラー: {e}")
            state["error_message"] = str(e)
            state["current_step"] = "split_text_error"
        
        return state
    
    def _enrich_metadata(self, state: ProcessingState) -> ProcessingState:
        """
        メタデータを付与する
        
        Args:
            state: 処理状態
            
        Returns:
            ProcessingState: 更新された処理状態
        """
        logger.info("=== ステップ4: メタデータ付与 ===")
        
        try:
            if state.get("error_message"):
                return state
            
            # LabelerAgentでメタデータ付与
            enriched_chunks = self.labeler.enrich_chunks(
                state["text_chunks"], 
                state["book_title"]
            )
            
            # 書籍タイトルを改善
            enhanced_title = self.labeler.enhance_book_title(
                state["book_title"], 
                enriched_chunks
            )
            
            logger.info(f"メタデータ付与完了: {len(enriched_chunks)}個のチャンク")
            
            state["enriched_chunks"] = enriched_chunks
            state["book_title"] = enhanced_title
            state["current_step"] = "enrich_metadata_complete"
            
        except Exception as e:
            logger.error(f"メタデータ付与エラー: {e}")
            state["error_message"] = str(e)
            state["current_step"] = "enrich_metadata_error"
        
        return state
    
    def _format_yaml(self, state: ProcessingState) -> ProcessingState:
        """
        YAML形式に整形する
        
        Args:
            state: 処理状態
            
        Returns:
            ProcessingState: 更新された処理状態
        """
        logger.info("=== ステップ5: YAML整形 ===")
        
        try:
            if state.get("error_message"):
                return state
            
            # YAMLFormatterAgentでYAML整形
            yaml_data = self.formatter.format_to_yaml(
                state["enriched_chunks"], 
                state["book_title"]
            )
            
            # メタデータを追加
            enhanced_yaml = self.formatter.add_metadata(yaml_data)
            
            logger.info("YAML整形完了")
            
            state["yaml_data"] = enhanced_yaml
            state["current_step"] = "format_yaml_complete"
            
        except Exception as e:
            logger.error(f"YAML整形エラー: {e}")
            state["error_message"] = str(e)
            state["current_step"] = "format_yaml_error"
        
        return state
    
    def _save_output(self, state: ProcessingState) -> ProcessingState:
        """
        出力ファイルを保存する
        
        Args:
            state: 処理状態
            
        Returns:
            ProcessingState: 更新された処理状態
        """
        logger.info("=== ステップ6: 出力保存 ===")
        
        try:
            if state.get("error_message"):
                return state
            
            # YAMLファイルとして保存
            success = self.formatter.save_to_file(self.output_file, state["yaml_data"])
            
            if success:
                logger.info(f"出力ファイル保存完了: {self.output_file}")
                state["processing_complete"] = True
                state["current_step"] = "save_output_complete"
            else:
                raise Exception("ファイル保存に失敗しました")
            
        except Exception as e:
            logger.error(f"出力保存エラー: {e}")
            state["error_message"] = str(e)
            state["current_step"] = "save_output_error"
        
        return state
    
    def process(self) -> bool:
        """
        書籍構造化処理を実行する
        
        Returns:
            bool: 処理成功フラグ
        """
        logger.info("書籍構造化分割ツールを開始します")
        
        # 初期状態を設定
        initial_state = ProcessingState(
            input_text="",
            book_title="",
            structure_analysis={},
            text_chunks=[],
            enriched_chunks=[],
            yaml_data={},
            current_step="initialized",
            error_message="",
            processing_complete=False
        )
        
        try:
            # ワークフローを実行
            final_state = self.workflow.invoke(initial_state)
            
            # 結果を確認
            if final_state.get("processing_complete"):
                logger.info("処理が正常に完了しました")
                self._print_summary(final_state)
                return True
            else:
                logger.error(f"処理が失敗しました: {final_state.get('error_message', '不明なエラー')}")
                return False
                
        except Exception as e:
            logger.error(f"ワークフロー実行中にエラーが発生しました: {e}")
            return False
    
    def _print_summary(self, final_state: ProcessingState):
        """
        処理結果のサマリーを出力する
        
        Args:
            final_state: 最終処理状態
        """
        print("\n" + "="*50)
        print("📚 書籍構造化分割ツール - 処理完了")
        print("="*50)
        
        print(f"📖 書籍タイトル: {final_state['book_title']}")
        print(f"📄 入力ファイル: {self.input_file}")
        print(f"💾 出力ファイル: {self.output_file}")
        
        if final_state.get("yaml_data"):
            chapters = final_state["yaml_data"].get("chapters", [])
            total_sections = sum(len(ch.get("sections", [])) for ch in chapters)
            
            print(f"📚 総章数: {len(chapters)}")
            print(f"📑 総節数: {total_sections}")
            
            if final_state["yaml_data"].get("metadata", {}).get("statistics"):
                stats = final_state["yaml_data"]["metadata"]["statistics"]
                print(f"📊 総文字数: {stats.get('total_content_length', 0):,}")
                # 推定読書時間の出力を削除
        
        print("="*50)
        print("✅ 処理が正常に完了しました！")


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='書籍構造化分割ツール')
    parser.add_argument('input_file', help='入力ファイルのパス')
    parser.add_argument('output_file', help='出力ファイルのパス')
    args = parser.parse_args()
    
    try:
        # ツールを初期化
        structurer = TextBookStructurer(args.input_file, args.output_file)
        
        # 処理を実行
        success = structurer.process()
        
        # 終了コードを設定
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("処理が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()