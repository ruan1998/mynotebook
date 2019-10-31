import numpy as np

def diaspora_storage(arr: np.ndarray(shape=(2,2)), uselib: bool=False) -> dict:
    '''
        将numpy矩阵元素以(row, col):value键值对方式存储在字典中。
    '''

    if not uselib:
        store_dic = {(ir, ic):arr[ir, ic] for ir, row in enumerate(arr) 
                                          for ic, col in enumerate(row) if col != 0}
    else:
        store_dic = {}
        index, ncols = np.flatnonzero(arr), arr.shape[1]
        for i in index:
            row = int(i/ncols)
            store_dic[(row, i-row*ncols)] = arr[row, i-row*ncols]


    return store_dic

example = np.eye(10000)
exam_dic = diaspora_storage(example, uselib=True)
print(exam_dic)

