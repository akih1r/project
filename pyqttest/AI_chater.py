




import os
import sys
import re
import google.generativeai as genai

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QMessageBox,
)


# -------------------------------
# Gemini SDK setup (no hard-coded key)
# -------------------------------
API_KEY = "AIzaSyDogoo-DjJv0PIzy2bOvNEpg2S5vsMqbpg"
genai.configure(api_key=API_KEY)
MODEL = genai.GenerativeModel("gemini-1.5-flash")


# -------------------------------
# Worker to call Gemini off the UI thread
# -------------------------------
class ApiWorker(QObject):
    finished = pyqtSignal(str, str)  #QObject を継承 → PyQtのオブジェクトとしてシグナルが使えるようになる
                                     #finished というシグナルを定義
                                     #引数は str, str
                                     #1つ目: AIからの返答テキスト    2つ目: エラーメッセージ（正常なら空文字）

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt #prompt はユーザーが入力したテキスト

    def run(self):
        try:
            resp = MODEL.generate_content(self.prompt) #Aiに送信
            # resp.text may be None if blocked or empty
            text = getattr(resp, "text", None)
            if not text:
                text = "[応答が空でした]"
            self.finished.emit(text, "") #成功した場合 → finished シグナルを発火（emit）
        except Exception as e: #eはえらーメッセージ
            self.finished.emit("", str(e))


# -------------------------------
# UI Widgets
# -------------------------------
class ChatComposer(QPlainTextEdit):
    """Shift+Enter で改行、Enter で送信する入力欄"""

    def __init__(self, on_send):
        super().__init__()
        self.on_send = on_send
        self.setPlaceholderText("メッセージを入力")
        self.setFixedHeight(60)
        
    
    

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                return super().keyPressEvent(e)  # 改行
            # Enter 単独は送信
            text = self.toPlainText().strip()
            if text:
                self.on_send(text)
                self.clear()
            return
        super().keyPressEvent(e)


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Chat")
        self.resize(420, 640)                    
        
        #Window[ Widget{ }]

        # 中央ウィジェット + 縦レイアウト
        root = QWidget()
        self.setCentralWidget(root)   #rootをメインのｗｉｄｇｅｔに設定このメソッドは継承したクラスのもの
        v = QVBoxLayout(root)
        v.setContentsMargins(8, 8, 8, 8) #外周余白
        v.setSpacing(8)                #子同士の間隔を８ｐｘ

        # 上: メッセージ表示
        self.chat = QListWidget()
        self.chat.setUniformItemSizes(False)    #各行の高さを同じにしない設定。折返しや複数行テキストがあると行ごとに高さが変わるので False が必須。True だと最後に計測した高さに揃えられ、長文がはみ出す。
        self.chat.setWordWrap(True)  #テキストを行内で折り返す。長文メッセージが横に伸びず、縦に増えるようになる。
        v.addWidget(self.chat, stretch=1)

        # 下: 入力バー
        bar = QHBoxLayout()
        self.composer = ChatComposer(self.send_message)
        self.send_btn = QPushButton("送信")
        self.send_btn.setAutoDefault(False)
        self.send_btn.clicked.connect(self._send_from_button)
        bar.addWidget(self.composer, stretch=1)
        bar.addWidget(self.send_btn)
        v.addLayout(bar)

        # API キー未設定なら送信を無効化
        if not API_KEY:
            self.add_message(
                f"{API_KEY} が未設定です。環境変数に API キーを設定してください。",
                sender="ai",
            )
            self.composer.setEnabled(False)
            self.send_btn.setEnabled(False)

        # 初期メッセージ
        self.add_message("こんにちは！下のバーは常に表示されます。", sender="ai")

        # Placeholders for current worker/thread to keep references
        self._thread: QThread | None = None
        self._worker: ApiWorker | None = None

    # ---- UI helpers ----
    def _send_from_button(self): #中身の文字列を Python の str として取り出す」メソッド
        text = self.composer.toPlainText().strip()
        if text:
            self.send_message(text)
            self.composer.clear()

    def add_message(self, text: str, sender: str): #チャットの吹き出しを1つ追加する」処理
        item = QListWidgetItem() #チャットリスト（QListWidget）に入れる「枠」
        bubble = QLabel(text)  #実際に表示するメッセージの文字列
        bubble.setWordWrap(True)  #長文のときに自動で折り返す
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse) #ユーザーがコピーできるようにテキスト選択を有効化

        if sender == "user":
            bubble.setStyleSheet(
                "background:#dcf8c6; padding:8px 10px; border-radius:10px; margin-left:60px;"
            )  #背景色を LINEっぽい緑 (#dcf8c6)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight)  #右寄せ（自分のメッセージ）
        else:
            bubble.setStyleSheet(
                "background:#ffffff; padding:8px 10px; border-radius:10px; margin-right:60px;"
            ) #背景色を白
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft) #左寄せ（相手のメッセージ）

        self.chat.addItem(item) #チャットリストに空のアイテムを追加
        self.chat.setItemWidget(item, bubble)#そのアイテムの中身を QLabel（吹き出し）で置き換える
        item.setSizeHint(bubble.sizeHint())
        self.chat.scrollToBottom() ##そのアイテムの中身を QLabel（吹き出し）で置き換える

    # ---- Chat logic ----
    def send_message(self, text: str):
        self.add_message(text, sender="user")
        # 非同期で Gemini を叩く
        self._start_api_call(text) #text->prompt

    def _start_api_call(self, prompt: str):
        # UI を一時無効化
        self.send_btn.setEnabled(False)
        self.composer.setEnabled(False)

        self._thread = QThread(self)
        self._worker = ApiWorker(prompt)   #aiに対してテキストをわたす
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)  #APIWOrkerをはしらせるスレッド開始時に ワーカーの処理をそのスレッド側で実行する。
        self._worker.finished.connect(self._on_api_finished)   #APIworker.run() 内で finished.emit(result, error) が呼ばれるとここが発火。ApiWorker.run() 内で finished.emit(result, error) が出たら UI更新用のスロットが呼ばれる（受け側＝メインスレッドなのでUI操作OK）
        self._worker.finished.connect(self._thread.quit)        #ワーカー処理が終わったらスレッドのイベントループを終了させる。
        self._worker.finished.connect(self._worker.deleteLater) #obj.deleteLater()と呼ぶと、そのオブジェクトに「次の安全なタイミングで削除してね」というイベントが投げられる
        self._thread.finished.connect(self._thread.deleteLater) #つまりfinishがえみっとで発火すると、スレッドとAPIWOrkeが消える

        self._thread.start()

    def _on_api_finished(self, reply_text: str, error: str):
        # UI を再度有効化
        self.send_btn.setEnabled(True)
        self.composer.setEnabled(True)
        self.composer.setFocus()

        if error:
            # よくあるエラー: 401/403 (認証), 429 (レート制限)
            self.add_message(f"エラー: {error}", sender="ai")
        else:
            self.add_message(reply_text, sender="ai")


def main():
    app = QApplication(sys.argv)
    w = ChatWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

