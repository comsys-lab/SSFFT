#!/usr/bin/python3

import os, sys

FFT_Length = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# FFT_Length = [7, 8, 9, 10, 11, 12]
# nFFTs = [1, 10, 100, 1000, 10000]
# nFFTs = [1000, 10000]
# nFFTs = [16384, 65536, 262144, 1048576, 4194304]
nFFTs = [16384]
# nSMs = [7, 0, 8, 9, 10]
nSMs = [1, 2, 3, 4, 6, 12]

file_name = "A1.txt"

for n in nFFTs:
    f = open(file_name, 'a')
    f.write(f"\n****** {n/1024} MB ******\n")
    f.close()
    for x in nSMs:
        f = open(file_name, 'a')
        if x == 0: 
            f.write(f"\n****** SSFFT_base ******\n")
        else:
            f.write(f"\n****** SSFFT {x} SMs ******\n")
        f.close()
        batch = n
        for j in FFT_Length:
            f = open(file_name, 'a')
            terminal_command = f"./SSFFT.exe {j} 1000 {x} {batch}" # (FFT length) (iteration) (nSMs) (nFFTs)
            print(terminal_command)
            stream = os.popen(terminal_command)
            f.write(stream.read())
            # f.write('\n')
            f.close()
            batch = int(batch/2)
        f = open(file_name, 'a')
        # f.write(f"\n*** End ***\n")
        f.close()
