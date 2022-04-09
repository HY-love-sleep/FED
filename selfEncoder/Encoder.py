def Encoder(lr, e, original_y, we, wd):
    '''

    :param lr:  学习率
    :param e:  训练伦次
    :param original_y:  初始标签 one-hot编码
    :param we:  编码器参数
    :param wd:  解码器参数
    :return:  [we, wd]
    '''
    for i in range(e):
        i += 1

