import argparse
import os
import re
import tiktoken

from bs4 import BeautifulSoup


class DocumentPreprocessor():
    def __init__(self, openai_model: str):
        self.openai_model = openai_model

    def create_chunks(self, content: str) -> list:
        """
        HTML/SGMLのデータからテキストを取り出して整形し、LLMが扱いやすいサイズに分割する
        """

        struct = BeautifulSoup(content, "html.parser")

        # <xref>の情報がget_text()で消えてしまうためテキストに置換しておく
        # 例: <xref linkend="app-pgdump"/> -> app-pgdump
        for xref in struct.find_all("xref"):
            linkend_text = xref.get("linkend")
            xref.replace_with(linkend_text if linkend_text else "")

        text = struct.get_text()

        # 行頭の空白を削除する
        text = "\n".join([line.lstrip() for line in text.split("\n")])

        # 3つ以上連続する改行を2つに置換する
        text = re.sub("\n{3,}", "\n\n", text)

        # LLMが扱いやすいサイズに分割する
        chunks = self._chunk_text(text)

        return chunks

    def _chunk_text(self, text: str) -> list:
        """
        テキストをLLMが扱いやすいサイズに分割する

        参考: https://github.com/Azure-Samples/azure-search-openai-demo/blob/main/scripts/prepdocslib/textsplitter.py
        - 区切り文字に日本語の句読点を追加
        - 文字数ではなくトークン数で数えるように変更
        """

        # 分割するトークン数の目安
        N_TOKENS_TARGET = 1000

        # 句読点を探す文字数の上限
        MAX_CHARS_SEARCH = 100

        # チャンク同士で重ねる文字数
        CHUNK_OVERLAP = 100

        SENTENCE_ENDINGS = [
            ".", "!", "?", "．", "。", "！", "？"
        ]

        WORDS_BREAKS = [
            ",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n",
            "，", "、", "；", "：", "　", "（", "）", "「", "」", "『", "』", "【", "】", "｛", "｝"
        ]

        chunks = []
        length = len(text)
        start = 0     # 切り取り開始位置
        end = length  # 切り取り終了位置

        while start + CHUNK_OVERLAP < length:
            """
            textからtext[start:end]を取り出す。
            start、endを仮決めして、startを前方、endを後方に句読点が見つかるまで広げていく
            """

            last_word = -1
            n_chars = self._calc_char_length_from_tokens(
                text[start:], N_TOKENS_TARGET)
            end = start + n_chars

            if end > length:
                end = length
            else:
                # start + n_charsのところから後方に向かって読点を探す。SEARCH_LIMITだけ探したらやめる
                while (
                    end < length
                    and end < start + n_chars + MAX_CHARS_SEARCH
                    and text[end] not in SENTENCE_ENDINGS
                ):
                    # 句点だった場合はメモしておく
                    if text[end] in WORDS_BREAKS:
                        last_word = end
                    end += 1

                if (
                    end < length
                    and text[end] not in SENTENCE_ENDINGS
                    and last_word > 0
                ):
                    # 読点が見つからなかったが句点は見つかった場合、句点を区切りにする
                    end = last_word

            if end < length:
                end += 1  # 位置を次の文の先頭にする

            last_word = -1
            start_origin = start

            # startのところから前方に向かって読点を探す。SEARCH_LIMITだけ探したらやめる
            while (
                start > 0
                and start > start_origin - MAX_CHARS_SEARCH
                and text[start] not in SENTENCE_ENDINGS
            ):
                if text[start] in WORDS_BREAKS:
                    # 句点だった場合はメモしておく
                    last_word = start
                start -= 1

            if (
                text[start] not in SENTENCE_ENDINGS
                and last_word > 0
            ):
                # 読点が見つからなかったが句点は見つかった場合、句点を区切りにする
                start = last_word

            if start > 0:
                start += 1  # 位置を次の文の先頭にする

            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP

        if start + CHUNK_OVERLAP < end:
            chunks.append(text[start:end])

        return chunks

    def _calc_char_length_from_tokens(self, text: str, n_tokens_target: int) -> int:
        """
        テキストの先頭から指定数のトークンを取り出すと何文字になるのかを求める
        """

        encoding = tiktoken.encoding_for_model(self.openai_model)
        low = 0
        high = len(text)

        # 二分探索
        while low <= high:
            mid = (low + high) // 2
            n_tokens = len(encoding.encode(text[:mid]))

            if n_tokens == n_tokens_target:
                return mid

            if n_tokens < n_tokens_target:
                low = mid + 1
            else:
                high = mid - 1

        # ぴったり合うとは限らない。近ければ良い
        return (low + high) // 2


if __name__ == "__main__":
    # テスト用
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to be processed")
    args = parser.parse_args()

    openai_model = os.environ.get(
        "AZURE_OPENAI_MODEL", "text-embedding-ada-002")
    preprocessor = DocumentPreprocessor(openai_model)

    print("Processing: " + args.file)

    with open(args.file, "r") as f:
        content = f.read()

    chunks = preprocessor.create_chunks(content)
    encoding = tiktoken.encoding_for_model(openai_model)

    for i, chunk in enumerate(chunks):
        print(str(i) + ":" + str(len(encoding.encode(chunk))) +
              ":" + chunk[:50].replace("\n", " "))

    print("\n========================================\n".join(chunks))
