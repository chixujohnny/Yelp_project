# coding:utf-8

# 将wiki的xml格式转换成text格式

import sys
import logging # 开启日志
import os.path


if __name__ == '__main__':

    program = os.path.basename(sys.argv[0]) # sys.argv是一个list, sys.argv[0]表示代码本身文件路径
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))