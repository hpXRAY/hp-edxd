from PyQt6.QtWidgets import QApplication, QMainWindow, QToolBar, QAction
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.toolbar = QToolBar(self)
        self.addToolBar(self.toolbar)

        action1 = QAction("Action 1", self)
        action2 = QAction("Action 2", self)

        self.toolbar.addAction(action1)
        self.toolbar.addAction(action2)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())