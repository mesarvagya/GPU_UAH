#Simple matrix multiply script matrixmul_test.sh
#Set up as bash shell as shown below
#!/bin/bash
# M:10x10 N:10x10 => P:10x10 test
./matrixmul_cu 10 > matrixmul.txt
rm *.gpu *.gold *.bin
# M:16x16 N:16x16 => P:16x16 Single block testing
./matrixmul_cu 16 > matrixmul.txt
# erase files generated to not exceed quota
rm *.gpu *.gold *.bin
# M:1024x2048 N:2048x1024 => P:1024x1024 test
./matrixmul_cu 1024 2048 1024 >> matrixmul.txt
# erase files generated to not exceed quota
rm *.gpu *.gold *.bin
