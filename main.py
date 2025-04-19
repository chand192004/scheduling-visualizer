import sys
from PyQt6.QtWidgets import QApplication
from ui import ProcessSchedulerUI

def main():
    app = QApplication(sys.argv)
    window = ProcessSchedulerUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 