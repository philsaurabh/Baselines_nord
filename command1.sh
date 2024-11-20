#python teacher_n.py --arch ResNet34  --gpu-id 2
# Student
python mstudent.py --t-path ./experimentse6/teacher6_ResNet34_seed0/ --s-arch ResNet34    --lr 0.01 --gpu-id 2
# Student
python mstudent.py --t-path ./experiments/teacher_ResNet34_seed0/ --s-arch ResNet34    --lr 0.01 --gpu-id 2