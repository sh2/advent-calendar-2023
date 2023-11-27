import argparse
import os
import re
import tiktoken

from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass
from document_preprocessor import DocumentPreprocessor


@dataclass
class Section:
    document: str
    chapter: str
    sect1: str
    sect2: str
    text: str


class DocumentPreprocessorCustom(DocumentPreprocessor):
    def __init__(self, openai_model: str, document_name: str):
        super().__init__(openai_model)
        self.document_name = document_name

    def create_chunks(self, content: str) -> list:
        """
        SGMLのデータから文書の構造を取り出して整形し、LLMが扱いやすいサイズに分割する
        """

        chunks = []
        section_list: list[Section] = self._extract_section_list(content)

        for section in section_list:
            text = section.text

            # 行頭の空白を削除する
            text = "\n".join([line.lstrip() for line in text.split("\n")])

            # 3つ以上連続する改行を2つに置換する
            text = re.sub("\n{3,}", "\n\n", text)

            # 分割する
            text_chunk_list = self._chunk_text(text)

            for text_chunk in text_chunk_list:
                text_temp = []
                text_temp.append(f"# {section.document}")

                if section.chapter:
                    text_temp.append(f"## {section.chapter}")

                if section.sect1:
                    text_temp.append(f"### {section.sect1}")

                if section.sect2:
                    text_temp.append(f"#### {section.sect2}")

                text_temp.append(text_chunk)
                chunks.append("\n".join(text_temp))

        return chunks

    def _extract_section_list(self, content: str) -> list:
        """
        SGMLのデータから<chapter><sect1><sect2>の構造を取り出す
        """

        section_list = []
        struct = BeautifulSoup(content, "html.parser")

        # <xref>の情報がget_text()で消えてしまうためテキストに置換しておく
        # 例: <xref linkend="app-pgdump"/> -> app-pgdump
        for xref in struct.find_all("xref"):
            linkend_text = xref.get("linkend")
            xref.replace_with(linkend_text if linkend_text else "")

        # <chapter>
        chapter_list: list[Tag] = struct.find_all(["chapter", "preface"])

        # <chapter>あるいは<preface>がなく、いきなり<sect1>から始まるファイルへの対応
        if chapter_list == []:
            chapter_list = [struct]

        for chapter in chapter_list:
            chapter_title = self._find_direct_child_title(chapter)
            chapter_text = chapter.get_text()

            # <sect1>
            sect1_list: list[Tag] = chapter.find_all("sect1")
            for sect1 in sect1_list:
                sect1_title = self._find_direct_child_title(sect1)
                sect1_text = sect1.get_text()

                # <chapter>の内側で<sect1>の外側にあるテキストを取り出す
                chapter_text_splitted = chapter_text.split(sect1_text)
                chapter_text_preface = chapter_text_splitted[0]
                chapter_text = sect1_text.join(chapter_text_splitted[1:])

                # <chapter>の序文があれば出力する
                if chapter_text_preface.split():
                    section_list.append(
                        Section(self.document_name, chapter_title, "", "", chapter_text_preface))

                # <sect2>
                sect2_list: list[Tag] = sect1.find_all("sect2")
                for sect2 in sect2_list:
                    sect2_title = self._find_direct_child_title(sect2)
                    sect2_text = sect2.get_text()

                    # <sect1>の内側で<sect2>の外側にあるテキストを取り出す
                    sect1_text_splitted = sect1_text.split(sect2_text)
                    sect1_text_preface = sect1_text_splitted[0]
                    sect1_text = sect2_text.join(sect1_text_splitted[1:])

                    # <sect1>の序文があれば出力する
                    if sect1_text_preface.split():
                        section_list.append(
                            Section(self.document_name, chapter_title, sect1_title, "", sect1_text_preface))

                    # <sect2>を出力する
                    section_list.append(Section(
                        self.document_name, chapter_title, sect1_title, sect2_title, sect2_text))

                # <sect1>の残りがあれば出力する
                if sect1_text.split():
                    section_list.append(
                        Section(self.document_name, chapter_title, sect1_title, "", sect1_text))

            # <chapter>の残りがあれば出力する
            if chapter_text.split():
                section_list.append(
                    Section(self.document_name, chapter_title, "", "", chapter_text))

        return section_list

    def _find_direct_child_title(self, struct: Tag) -> str:
        """
        タグの直下にあるタイトルのテキストを取得する

        <tag1>
            <title>このテキストを取得する</title>
            <tag2>
                <title>このテキストは取得しない</title>
            </tag2>
        </tag1>
        """

        title = struct.find("title")

        if isinstance(title, Tag):
            parent = title.find_parent()

            if parent is not None and parent.name == struct.name:
                return title.get_text()

        return ""


if __name__ == "__main__":
    # テスト用
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to be processed")
    args = parser.parse_args()

    openai_model = os.environ.get(
        "AZURE_OPENAI_MODEL", "text-embedding-ada-002")
    document_name = os.environ.get("DOCUMENT_NAME", "Document")
    preprocessor = DocumentPreprocessorCustom(openai_model, document_name)

    print("Processing: " + args.file)

    with open(args.file, "r") as f:
        content = f.read()

    chunks = preprocessor.create_chunks(content)
    encoding = tiktoken.encoding_for_model(openai_model)

    for i, chunk in enumerate(chunks):
        print(str(i) + ":" + str(len(encoding.encode(chunk))) +
              ":" + chunk[:200].replace("\n", " "))

    print("\n========================================\n".join(chunks))
