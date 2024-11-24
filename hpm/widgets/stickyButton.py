# -*- coding: utf8 -*-

# DISCLAIMER
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from PyQt6 import QtWidgets, QtCore

class StickyButton(QtWidgets.QPushButton):
    clicked_signal = QtCore.pyqtSignal()
    pressed_signal = QtCore.pyqtSignal()
    repeat_signal = QtCore.pyqtSignal()
    release_signal = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        QtWidgets.QPushButton.__init__(self, *args, **kwargs)
        self.setAutoRepeat(True)
        self.setAutoRepeatDelay(1000)
        self.setAutoRepeatInterval(1000)
        self.clicked.connect(self.handleClicked)
        self._state = 0

    def handleClicked(self):
        if self.isDown():
            if self._state == 0:
                self._state = 1
                self.setAutoRepeatInterval(100)
                self.pressed_signal.emit()
            else:
                self.repeat_signal.emit()
        elif self._state == 1:
            self._state = 0
            self.setAutoRepeatInterval(1000)
            self.release_signal.emit()
        else:
            self.clicked_signal.emit()

if __name__ == '__main__':

    import sys
    app = QtWidgets.QApplication(sys.argv)
    button = StickyButton('Test Button')
    button.show()
    sys.exit(app.exec())