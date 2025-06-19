#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LabelerAgent - メタ付与エージェント

タイトル補完、要約生成などのメタデータを付与するエージェント
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from openai_client import OpenAIClient
from utils.helpers import generate_summary, clean_text
from agents.splitter import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class EnrichedChunk:
    """メタデータが付与されたチャンクを表すデータクラス"""
    type: str
    number: str
    title: str
    content: str
    summary: str = ""
    keywords: List[str] = None
    start_pos: int = 0
    end_pos: int = 0
    parent_chapter: Optional[int] = None
    word_count: int = 0
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class LabelerAgent:
    """
    メタ付与エージェント
    
    テキストチャンクにタイトル補完、要約生成、キーワード抽出などの
    メタデータを付与する
    """
    
    def __init__(self, openai_client: OpenAIClient):
        """
        初期化
        
        Args:
            openai_client: OpenAIクライアント
        """
        self.client = openai_client
        self.enriched_chunks = []
        
    def enrich_chunks(self, chunks: List[TextChunk], book_title: str = "タイトル未設定") -> List[EnrichedChunk]:
        """
        チャンクにメタデータを付与する
        
        Args:
            chunks: テキストチャンクリスト
            book_title: 書籍タイトル
            
        Returns:
            List[EnrichedChunk]: メタデータ付与済みチャンクリスト
        """
        logger.info("チャンクのメタデータ付与を開始します")
        
        enriched_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"チャンク {i+1}/{len(chunks)} を処理中: {chunk.type} {chunk.number}")
            
            # 基本情報をコピー
            enriched = EnrichedChunk(
                type=chunk.type,
                number=chunk.number,
                title=chunk.title,
                content=chunk.content,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                parent_chapter=chunk.parent_chapter,
                word_count=chunk.word_count
            )
            
            # メタデータを生成
            self._generate_metadata(enriched, book_title, i, len(chunks))
            
            enriched_chunks.append(enriched)
        
        self.enriched_chunks = enriched_chunks
        logger.info(f"メタデータ付与完了: {len(enriched_chunks)}個のチャンク")
        
        return enriched_chunks
    
    def _generate_metadata(self, chunk: EnrichedChunk, book_title: str, index: int, total: int):
        """
        チャンクのメタデータを生成する
        
        Args:
            chunk: エンリッチ対象チャンク
            book_title: 書籍タイトル
            index: チャンクのインデックス
            total: 総チャンク数
        """
        # タイトルの改善
        chunk.title = self._improve_title(chunk)
        
        # 要約の生成
        chunk.summary = self._generate_summary(chunk, book_title)
        
        # キーワードの抽出
        chunk.keywords = self._extract_keywords(chunk)
    
    def _improve_title(self, chunk: EnrichedChunk) -> str:
        """
        AIを使用してタイトルを改善する
        
        Args:
            chunk: エンリッチ対象チャンク
            
        Returns:
            str: 改善されたタイトル
        """
        # タイトルが既に適切な場合はそのまま返す
        if len(chunk.title) > 5 and not chunk.title.startswith("章") and not chunk.title.startswith("節"):
            return chunk.title
        
        # コンテンツが短い場合は簡単な処理
        if len(chunk.content) < 100:
            return chunk.title
        
        system_prompt = """あなたは書籍編集の専門家です。
与えられたテキストセクションの内容に基づいて、適切で魅力的なタイトルを生成してください。

要件：
- 50文字程度（最大でも50文字以内）
- 内容を的確に表現
- 読者の興味を引く
- 日本語で自然な表現
- 体言止め（「〜の考察」「〜への道」など名詞で終わる形式）を基本とする
- 簡潔で力強い印象を与えるタイトルにする

タイトルのみを返してください。説明は不要です。"""
        
        # 分析用テキスト（先頭500文字）
        analysis_text = chunk.content[:500] if len(chunk.content) > 500 else chunk.content
        
        user_prompt = f"""以下のセクションに適切なタイトルを付けてください：

現在のタイトル: {chunk.title}
セクション番号: {chunk.number}
セクションタイプ: {chunk.type}

内容:
{analysis_text}

重要: タイトルは体言止め（名詞で終わる形式）を基本とし、簡潔で力強い印象を与えるようにしてください。例えば「〜の考察」「〜への道」「〜の真実」などの形式が望ましいです。
"""
        
        try:
            response = self.client.ask(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=50,
                temperature=0.4
            )
            
            improved_title = self.client.extract_text(response).strip()
            
            # 結果の検証
            if improved_title and len(improved_title) <= 50:
                logger.debug(f"タイトル改善: '{chunk.title}' -> '{improved_title}'")
                return improved_title
            else:
                logger.warning(f"タイトル改善結果が不適切: {improved_title}")
                return chunk.title
                
        except Exception as e:
            logger.error(f"タイトル改善中にエラーが発生しました: {e}")
            return chunk.title
    
    def _generate_summary(self, chunk: EnrichedChunk, book_title: str) -> str:
        """
        AIを使用して要約を生成する
        
        Args:
            chunk: エンリッチ対象チャンク
            book_title: 書籍タイトル
            
        Returns:
            str: 生成された要約
        """
        # コンテンツが短い場合は簡単な要約
        if len(chunk.content) < 200:
            return generate_summary(chunk.content, 100)
        
        system_prompt = """あなたは書籍要約の専門家です。
与えられたテキストセクションの要約を作成してください。

要件：
- 100文字以内
- 主要なポイントを含む
- 読みやすい日本語
- セクションの核心を捉える
- 元のテキストと同じ文体を使用する（「である調」「ですます調」など、元の文章の文体や語尾を維持）
- 元のテキスト内に複数の文体が混在している場合は、主要な文体を使用する

要約のみを返してください。説明は不要です。"""
        
        # 分析用テキスト（先頭1000文字）
        analysis_text = chunk.content[:1000] if len(chunk.content) > 1000 else chunk.content
        
        user_prompt = f"""以下のセクションの要約を作成してください：

書籍: {book_title}
セクション: {chunk.title} ({chunk.number})

内容:
{analysis_text}

重要: 元のテキストと同じ文体を維持してください。例えば、元のテキストが「である調」なら「である調」で、「ですます調」なら「ですます調」で要約してください。引用部分ではなく、主要な本文の文体に合わせてください。
"""
        
        try:
            response = self.client.ask(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=150,
                temperature=0.3
            )
            
            summary = self.client.extract_text(response).strip()
            
            # 結果の検証
            if summary and len(summary) <= 200:
                return summary
            else:
                logger.warning(f"要約生成結果が不適切: {len(summary)}文字")
                return generate_summary(chunk.content, 100)
                
        except Exception as e:
            logger.error(f"要約生成中にエラーが発生しました: {e}")
            return generate_summary(chunk.content, 100)
    
    def _extract_keywords(self, chunk: EnrichedChunk) -> List[str]:
        """
        AIを使用してキーワードを抽出する
        
        Args:
            chunk: エンリッチ対象チャンク
            
        Returns:
            List[str]: 抽出されたキーワードリスト
        """
        # コンテンツが短い場合はスキップ
        if len(chunk.content) < 100:
            return []
        
        system_prompt = """あなたはテキスト分析の専門家です。
与えられたテキストから重要なキーワードを抽出してください。

要件：
- 3-5個のキーワード
- 内容を代表する重要な語句
- 固有名詞や専門用語を優先
- カンマ区切りで出力

キーワードのみを返してください。説明は不要です。"""
        
        # 分析用テキスト（先頭800文字）
        analysis_text = chunk.content[:800] if len(chunk.content) > 800 else chunk.content
        
        user_prompt = f"""以下のテキストからキーワードを抽出してください：

タイトル: {chunk.title}
内容:
{analysis_text}
"""
        
        try:
            response = self.client.ask(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=100,
                temperature=0.2
            )
            
            keywords_text = self.client.extract_text(response).strip()
            
            # キーワードを分割してクリーンアップ
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            keywords = keywords[:5]  # 最大5個まで
            
            return keywords
                
        except Exception as e:
            logger.error(f"キーワード抽出中にエラーが発生しました: {e}")
            return []
    
    def enhance_book_title(self, original_title: str, chunks: List[EnrichedChunk]) -> str:
        """
        書籍全体の内容に基づいてタイトルを改善する
        
        Args:
            original_title: 元のタイトル
            chunks: エンリッチ済みチャンクリスト
            
        Returns:
            str: 改善されたタイトル
        """
        if original_title != "タイトル未設定" and len(original_title) > 5:
            return original_title
        
        # 全チャンクの要約を結合
        all_summaries = [chunk.summary for chunk in chunks if chunk.summary]
        combined_summary = " ".join(all_summaries[:5])  # 最初の5つの要約
        
        # 全キーワードを収集
        all_keywords = []
        for chunk in chunks:
            all_keywords.extend(chunk.keywords)
        
        # 頻出キーワードを特定
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_keywords_text = ", ".join([kw[0] for kw in top_keywords])
        
        system_prompt = """あなたは書籍タイトル作成の専門家です。
与えられた情報に基づいて、魅力的で適切な書籍タイトルを生成してください。

要件：
- 20文字以内
- 内容を的確に表現
- 読者の興味を引く
- 覚えやすい

タイトルのみを返してください。説明は不要です。"""
        
        user_prompt = f"""以下の情報に基づいて書籍タイトルを生成してください：

現在のタイトル: {original_title}
章数: {len([c for c in chunks if c.type == 'chapter'])}
主要キーワード: {top_keywords_text}

内容要約:
{combined_summary[:500]}
"""
        
        try:
            response = self.client.ask(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=50,
                temperature=0.4
            )
            
            enhanced_title = self.client.extract_text(response).strip()
            
            if enhanced_title and len(enhanced_title) <= 30:
                logger.info(f"書籍タイトル改善: '{original_title}' -> '{enhanced_title}'")
                return enhanced_title
            else:
                return original_title
                
        except Exception as e:
            logger.error(f"書籍タイトル改善中にエラーが発生しました: {e}")
            return original_title
    
    def get_enrichment_summary(self) -> Dict[str, Any]:
        """
        メタデータ付与結果の要約を取得する
        
        Returns:
            Dict[str, Any]: 付与要約
        """
        if not self.enriched_chunks:
            return {'total_chunks': 0}
        
        # 統計情報を計算
        total_summaries = len([c for c in self.enriched_chunks if c.summary])
        total_keywords = sum(len(c.keywords) for c in self.enriched_chunks)
        
        return {
            'total_chunks': len(self.enriched_chunks),
            'chunks_with_summaries': total_summaries,
            'total_keywords': total_keywords,
            'average_keywords_per_chunk': total_keywords / len(self.enriched_chunks) if self.enriched_chunks else 0
        }
        
    async def _generate_metadata_async(self, chunk: EnrichedChunk, book_title: str, index: int, total: int):
        """
        チャンクのメタデータを非同期で生成する
        
        Args:
            chunk: エンリッチ対象チャンク
            book_title: 書籍タイトル
            index: チャンクのインデックス
            total: 総チャンク数
        """
        # タイトルの改善
        improved_title_response = await self.client.ask_async(
            prompt=f"""以下のセクションに適切なタイトルを付けてください：

現在のタイトル: {chunk.title}
セクション番号: {chunk.number}
セクションタイプ: {chunk.type}

内容:
{chunk.content[:500] if len(chunk.content) > 500 else chunk.content}

重要: 元のテキストと同じ文体を維持してください。例えば、元のテキストが「である調」なら「である調」で、「ですます調」なら「ですます調」でタイトルを付けてください。引用部分ではなく、主要な本文の文体に合わせてください。
""",
            system="""あなたは書籍編集の専門家です。
与えられたテキストセクションの内容に基づいて、適切で魅力的なタイトルを生成してください。

要件：
- 15文字以内
- 内容を的確に表現
- 読者の興味を引く
- 日本語で自然な表現
- 元のテキストと同じ文体を使用する（「である調」「ですます調」など、元の文章の文体や語尾を維持）

タイトルのみを返してください。説明は不要です。""",
            max_tokens=50,
            temperature=0.4
        )
        
        # 要約の生成
        summary_response = await self.client.ask_async(
            prompt=f"""以下のセクションの要約を作成してください：

書籍: {book_title}
セクション: {chunk.title} ({chunk.number})

内容:
{chunk.content[:1000] if len(chunk.content) > 1000 else chunk.content}

重要: 元のテキストと同じ文体を維持してください。例えば、元のテキストが「である調」なら「である調」で、「ですます調」なら「ですます調」で要約してください。引用部分ではなく、主要な本文の文体に合わせてください。
""",
            system="""あなたは書籍要約の専門家です。
与えられたテキストセクションの要約を作成してください。

要件：
- 100文字以内
- 主要なポイントを含む
- 読みやすい日本語
- セクションの核心を捉える
- 元のテキストと同じ文体を使用する（「である調」「ですます調」など、元の文章の文体や語尾を維持）
- 元のテキスト内に複数の文体が混在している場合は、主要な文体を使用する

要約のみを返してください。説明は不要です。""",
            max_tokens=150,
            temperature=0.3
        )
        
        # キーワードの抽出
        keywords_response = await self.client.ask_async(
            prompt=f"""以下のテキストからキーワードを抽出してください：

タイトル: {chunk.title}
内容:
{chunk.content[:800] if len(chunk.content) > 800 else chunk.content}
""",
            system="""あなたはテキスト分析の専門家です。
与えられたテキストから重要なキーワードを抽出してください。

要件：
- 3-5個のキーワード
- 内容を代表する重要な語句
- 固有名詞や専門用語を優先
- カンマ区切りで出力

キーワードのみを返してください。説明は不要です。""",
            max_tokens=100,
            temperature=0.2
        )
        
        # 結果を処理
        improved_title = self.client.extract_text(improved_title_response).strip()
        if improved_title and len(improved_title) <= 30:
            chunk.title = improved_title
        
        summary = self.client.extract_text(summary_response).strip()
        if summary and len(summary) <= 200:
            chunk.summary = summary
        else:
            chunk.summary = generate_summary(chunk.content, 100)
        
        keywords_text = self.client.extract_text(keywords_response).strip()
        keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
        chunk.keywords = keywords[:5]  # 最大5個まで

    async def enrich_chunks_async(self, chunks: List[TextChunk], book_title: str = "タイトル未設定") -> List[EnrichedChunk]:
        """
        チャンクにメタデータを非同期で付与する
        
        Args:
            chunks: テキストチャンクリスト
            book_title: 書籍タイトル
            
        Returns:
            List[EnrichedChunk]: メタデータ付与済みチャンクリスト
        """
        logger.info("チャンクのメタデータ付与を開始します（非同期処理）")
        
        enriched_chunks = []
        
        # 基本情報をコピー
        for chunk in chunks:
            enriched = EnrichedChunk(
                type=chunk.type,
                number=chunk.number,
                title=chunk.title,
                content=chunk.content,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                parent_chapter=chunk.parent_chapter,
                word_count=chunk.word_count
            )
            enriched_chunks.append(enriched)
        
        # 非同期タスクを作成
        tasks = []
        for i, enriched in enumerate(enriched_chunks):
            task = self._generate_metadata_async(enriched, book_title, i, len(chunks))
            tasks.append(task)
        
        # 並行処理を実行
        await asyncio.gather(*tasks)
        
        self.enriched_chunks = enriched_chunks
        logger.info(f"メタデータ付与完了: {len(enriched_chunks)}個のチャンク")
        
        return enriched_chunks

    def enrich_chunks(self, chunks: List[TextChunk], book_title: str = "タイトル未設定") -> List[EnrichedChunk]:
        """
        チャンクにメタデータを付与する（同期版）
        
        Args:
            chunks: テキストチャンクリスト
            book_title: 書籍タイトル
            
        Returns:
            List[EnrichedChunk]: メタデータ付与済みチャンクリスト
        """
        return asyncio.run(self.enrich_chunks_async(chunks, book_title))