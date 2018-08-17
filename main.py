from STARGAN import *
import click
import argparse


desc = "Tensorflow config for StarGAN"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--phase', type=str, default='train', help='train or test ?')
parser.add_argument('--features_A', type=str, nargs='+', help='selected attributes for the one dataset',default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
parser.add_argument('--features_B', type=str, nargs='+', help='selected attributes for the another dataset',default=['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral'])
parser.add_argument('--txt1_dir', type=str, default='./data/train.txt', help='The path to the document for one dataset description')
parser.add_argument('--txt2_dir', type=str, default='./data/trainano.txt', help='The path to the document for another dataset description')
parser.add_argument('--imfile1', type=str, default='./data/train/', help='The folder directory where the training A picture is located')
parser.add_argument('--imfile2', type=str, default='./data/trainano/', help='The folder directory where the training B picture is located')
parser.add_argument('--save_path', type=str, default='./data/Trainims/', help='Image save path')
parser.add_argument('--ckpt_path', type=str, default='./data/ckpt/', help='checkpoint save path')
parser.add_argument('--logdir', type=str, default='./data/Log/', help='events save path')
parser.add_argument('--test_ckpt', type=str, default='./data/ckpt/', help='checkpoint to load')
parser.add_argument('--test_imsdir', type=str, default='./data/Test/', help='Test image path')
parser.add_argument('--test_save_path', type=str, default='./data/Testims/', help='New image file directory')
parser.add_argument('--test_target', type=str, default='./data/test_target.txt', help='Test target list')
    
parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
   
parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
   
parser.add_argument('--save_iters', type=int, default=20, help='The number of save images')
    
parser.add_argument('--ld', type=float, default=1, help='GP')
   
parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
   
parser.add_argument('--adv_weight', type=float, default=1, help='Adversarial loss weight')
   
parser.add_argument('--cls_weight', type=float, default=10, help='Class loss weight')
parser.add_argument('--recy_weight', type=float, default=10, help='Cyclic coherence loss weight')
args = parser.parse_args()




if __name__ == '__main__':
    if args.phase == 'train':
        print("Training finished!")
        gan = StarGan(args)
        gan.train()       
    if args.phase == 'test':
        print("Test finished!")
        gan.test()
