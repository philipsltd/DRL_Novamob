import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/filipe/thesis/drl_novamob/install/novamob_gym'
