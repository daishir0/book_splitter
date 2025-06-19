# book_splitter

## Overview
book_splitter is a tool for analyzing and structuring text books. It uses AI to understand the meaning of text, automatically detect chapters and sections, and output them in a structured format. The tool consists of two main components:

1. **TextBookStructurer**: Analyzes text files, detects structure, and outputs YAML format
2. **YAML Splitter**: Splits the YAML file into individual text files for each chapter/section

This tool is useful for authors, editors, and researchers who need to analyze and restructure large text documents.

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Steps
1. Clone the repository:
```bash
git clone https://github.com/daishir0/book_splitter
cd book_splitter
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Configure your OpenAI API key in `config.py`:
```python
OPENAI_API_KEY = "your-api-key-here"
```

## Usage

### TextBookStructurer
The main component analyzes text files and outputs structured YAML:

```bash
python main.py input_file output_file
```

Example:
```bash
python main.py my_book.txt structured_book.yaml
```

### YAML Splitter
Splits the YAML file into individual text files:

```bash
python yaml_splitter.py yaml_file
```

Example:
```bash
python yaml_splitter.py structured_book.yaml
```

This will create numbered text files in the same directory as the YAML file, with filenames in the format: `000001Title_Summary.txt`

## Notes
- Processing large text files may take time due to API calls
- The quality of structure detection depends on the clarity of the original text
- An OpenAI API key is required and API usage will incur costs
- The tool works best with well-structured text with clear chapter/section divisions

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

# book_splitter

## 概要
book_splitterは、テキスト形式の書籍を解析し構造化するツールです。AIを活用してテキストの意味を理解し、章や節を自動的に検出して構造化された形式で出力します。このツールは主に2つのコンポーネントで構成されています：

1. **TextBookStructurer**：テキストファイルを解析し、構造を検出してYAML形式で出力
2. **YAML Splitter**：YAMLファイルを章・節ごとの個別のテキストファイルに分割

このツールは、大量のテキスト文書を分析・再構成する必要がある著者、編集者、研究者に役立ちます。

## インストール方法

### 前提条件
- Python 3.8以上
- OpenAI APIキー

### 手順
1. リポジトリをクローンします：
```bash
git clone https://github.com/daishir0/book_splitter
cd book_splitter
```

2. 必要なパッケージをインストールします：
```bash
pip install -r requirements.txt
```

3. `config.py`にOpenAI APIキーを設定します：
```python
OPENAI_API_KEY = "あなたのAPIキーをここに入力"
```

## 使い方

### TextBookStructurer
メインコンポーネントはテキストファイルを解析し、構造化されたYAMLを出力します：

```bash
python main.py 入力ファイル 出力ファイル
```

例：
```bash
python main.py my_book.txt structured_book.yaml
```

### YAML Splitter
YAMLファイルを個別のテキストファイルに分割します：

```bash
python yaml_splitter.py YAMLファイル
```

例：
```bash
python yaml_splitter.py structured_book.yaml
```

これにより、YAMLファイルと同じディレクトリに、`000001タイトル_サマリー.txt`という形式の連番付きテキストファイルが作成されます。

## 注意点
- 大きなテキストファイルの処理はAPI呼び出しのため時間がかかる場合があります
- 構造検出の品質は、元のテキストの明確さに依存します
- OpenAI APIキーが必要で、API使用には費用が発生します
- このツールは、章や節の区切りが明確な、よく構造化されたテキストで最も効果を発揮します

## ライセンス
このプロジェクトはMITライセンスの下でライセンスされています。詳細はLICENSEファイルを参照してください。