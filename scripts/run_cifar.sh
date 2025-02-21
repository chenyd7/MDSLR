# kd, random
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0 -b 0.9 --selectway random --ratio 40 --trial 1 
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0 -b 0.9 --selectway random --ratio 40 --trial 2 
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0 -b 0.9 --selectway random --ratio 40 --trial 3 

# kd, medium, logit reshape
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0 -b 0.9 --selectway medium --std 1 --ratio 40 --modelsp 1 --trial 1 
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0 -b 0.9 --selectway medium --std 1 --ratio 40 --modelsp 1 --trial 2 
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0 -b 0.9 --selectway medium --std 1 --ratio 40 --modelsp 1 --trial 3 

# kd, medium, w/o logit reshape
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0 -b 0.9 --selectway medium --ratio 40 --modelsp 1 --trial 1 
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0 -b 0.9 --selectway medium --ratio 40 --modelsp 1 --trial 2 
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0 -b 0.9 --selectway medium --ratio 40 --modelsp 1 --trial 3 

# pefd, random
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill pefd --model_s MobileNetV2 -a 0 -b 25 --selectway random --ratio 40 --trial 1 
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill pefd --model_s MobileNetV2 -a 0 -b 25 --selectway random --ratio 40 --trial 2 
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill pefd --model_s MobileNetV2 -a 0 -b 25 --selectway random --ratio 40 --trial 3 

# pefd, medium, w/o logit reshape
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill pefd --model_s MobileNetV2 -a 0 -b 25 --selectway medium --ratio 40 --modelsp 1 --trial 1 
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill pefd --model_s MobileNetV2 -a 0 -b 25 --selectway medium --ratio 40 --modelsp 1 --trial 2 
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill pefd --model_s MobileNetV2 -a 0 -b 25 --selectway medium --ratio 40 --modelsp 1 --trial 3 

