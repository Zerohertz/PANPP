# read -p "Name of Directory: " name
# rm -rf ${name}.csv

# cd CLEval_1024
# python prepare.py --path=${name}
# python script.py --path=${name}

name=torch
cd CLEval_1024
python prepare.py --path=${name}
python script.py --path=${name}
cd ..

name=trt
cd CLEval_1024
python prepare.py --path=${name}
python script.py --path=${name}
cd ..

name=pth
cd CLEval_1024
python prepare.py --path=${name}
python script.py --path=${name}
cd ..