"""允许通过 python monitor 执行。"""

import os
import sys


_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
	sys.path.insert(0, _project_dir)

from monitor.main import main


main()
