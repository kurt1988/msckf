#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    datapath = './output.csv'
    data_frame = pd.read_csv(datapath, delimiter=',', header=0)
    frameID = np.array(data_frame['frameID']).astype(np.float64)
    position_err = np.array(data_frame['p_error']).astype(np.float64)

    print('mean / max position error: ', np.mean(position_err), np.max(position_err))

    plt.plot(frameID, position_err, 'b.')
    plt.xlabel('frameID')
    plt.ylabel('position error')
    plt.grid()
    plt.show()



