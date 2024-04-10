import time
from progressbar import progressbar

for i in progressbar(range(100)):
    print('Some text', i)
    time.sleep(0.1)