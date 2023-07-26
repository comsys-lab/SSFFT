#!/usr/bin/python3

import os, sys

# FFT_Length = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
FFT_Length = [7, 8, 9, 10, 11, 12]
# nFFTs = [1, 10, 100, 1000, 10000]
# nFFTs = [1000, 10000]
nFFTs = [100]
# FFT_Length = [128, 256, 512 ,1024 ,2048 ,4096 ,8192 ,16384 ,32768, 65536, 131072 ,262144 ,524288 ,1048576 ,2097152]
# FFT_Length = [[16384,1024]]
# nSMs = [7, 0, 8, 9, 10]
nSMs = [0]

file_name = "analysis1.txt"
for n in nFFTs:
    f = open(file_name, 'a')
    f.write(f"\n****** Batch {n} FFTs ******\n")
    f.close()
    for x in nSMs:
        f = open(file_name, 'a')
        if x == 0: 
            f.write(f"\n****** SSFFT_base ******\n")
        else:
            f.write(f"\n****** SSFFT {x} SMs ******\n")
        f.close()
        for j in FFT_Length:
            f = open(file_name, 'a')
            if n == 100 and x == 21:
                terminal_command = f"./SSFFT.exe {j} 1000 {x} 90" # (FFT length) (iteration) (nSMs) (nFFTs)
            else:
                terminal_command = f"./SSFFT.exe {j} 1 {x} {n}" # (FFT length) (iteration) (nSMs) (nFFTs)
            stream = os.popen(terminal_command)
            f.write(stream.read())
            # f.write('\n')
            f.close()
        f = open(file_name, 'a')
        # f.write(f"\n*** End ***\n")
        f.close()
