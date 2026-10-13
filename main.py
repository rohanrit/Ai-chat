import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

from src.gui import LocalAIGUI

if __name__ == "__main__":
    app = LocalAIGUI()
    app.mainloop()
