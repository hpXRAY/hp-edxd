
import sys
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton

class MultiChoiceDialog(QDialog):
    def __init__(self, choices):
        super().__init__()

        layout = QVBoxLayout()

        self.choice = None
        self.buttons = []

        for i, choice in enumerate(choices):
            button = QPushButton(choice)
            button.clicked.connect(lambda _, i=i: self.set_choice(i+1))
            layout.addWidget(button)
            self.buttons.append(button)

        self.setLayout(layout)

    def set_choice(self, choice):
        self.choice = choice
        self.accept()

def display_multi_choice_dialog(choices):
    dialog = MultiChoiceDialog(choices)
    if dialog.exec():
        return dialog.choice
    return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    choices = ["Choice 1", "Choice 2", "Choice 3"]  # Example choices, you can generate these dynamically
    response = display_multi_choice_dialog(choices)
    if response is not None:
        print("Chosen option:", response)
    else:
        print("No option chosen")
