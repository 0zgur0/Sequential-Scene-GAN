import argparse
import os, time
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_loader_fg_baseline import CocoData
from utils import  show_result
from networks import Discriminator, Generator_Baseline
from Feature_Matching import VGGLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='log', help='Name of the log folder')
    parser.add_argument('--save_models', type=bool, default=True, help='Set True if you want to save trained models')
    parser.add_argument('--pre_trained_model_path', type=str, default=None, help='Pre-trained model path')
    parser.add_argument('--pre_trained_model_epoch', type=str, default=None, help='Pre-trained model epoch e.g 200')
    parser.add_argument('--train_imgs_path', type=str, default='C:/Users/motur/coco/images/train2017', help='Path to training images')
    parser.add_argument('--train_annotation_path', type=str, default='C:/Users/motur/coco/annotations/instances_train2017.json', help='Path to annotation file, .json file')
    parser.add_argument('--category_names', type=str, default='giraffe,elephant,zebra,sheep,cow,bear',help='List of categories in MS-COCO dataset')
    parser.add_argument('--num_test_img', type=int, default=16,help='Number of images saved during training')
    parser.add_argument('--img_size', type=int, default=128,help='Generated image size')
    parser.add_argument('--local_patch_size', type=int, default=128, help='Image size of instance images after interpolation')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size')
    parser.add_argument('--train_epoch', type=int, default=400,help='Maximum training epoch')
    parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate')
    parser.add_argument('--optim_step_size', type=int, default=80,help='Learning rate decay step size')
    parser.add_argument('--optim_gamma', type=float, default=0.5,help='Learning rate decay ratio')
    parser.add_argument('--critic_iter', type=int, default=5,help='Number of discriminator update against each generator update')
    parser.add_argument('--noise_size', type=int, default=128,help='Noise vector size')
    parser.add_argument('--lambda_FM', type=float, default=1,help='Trade-off param for feature matching loss')
    parser.add_argument('--num_res_blocks', type=int, default=5,help='Number of residual block in generator network')
    parser.add_argument('--trade_off_G', type=float, default=0.1,help='Trade-off parameter which controls gradient flow to generator from D_local and D_glob')
    
    opt = parser.parse_args()
    print(opt)
       
    #Create log folder
    root = 'result_base/'
    model = 'coco_model_'
    result_folder_name = 'images_' + opt.log_dir
    model_folder_name = 'models_' + opt.log_dir
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + result_folder_name):
        os.mkdir(root + result_folder_name)
    if not os.path.isdir(root + model_folder_name):
        os.mkdir(root + model_folder_name)
    
    #Save the script
    copyfile(os.path.basename(__file__), root + result_folder_name + '/' + os.path.basename(__file__))

    #Define transformation for dataset images - e.g scaling
    transform = transforms.Compose([transforms.Scale((opt.img_size,opt.img_size)),
                                    transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 
    #Load dataset
    category_names = opt.category_names.split(',')
    dataset = CocoData(root = opt.train_imgs_path,annFile = opt.train_annotation_path,category_names = category_names,
                            transform=transform,final_img_size=opt.img_size)


    #Discard images contain very small instances  
    dataset.discard_small(min_area=0.0, max_area= 1)
    dataset.discard_bad_examples('bad_examples_list.txt')
    
    #Define data loader
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    #For evaluation define fixed masks and noises
    data_iter = iter(train_loader)
    sample_batched = data_iter.next() 
    y_fixed = sample_batched['seg_mask'][0:opt.num_test_img]
    y_fixed = Variable(y_fixed.cuda())   
    z_fixed = torch.randn((opt.num_test_img, opt.noise_size))
    z_fixed= Variable(z_fixed.cuda())
        
    #Define networks
    G_base = Generator_Baseline(z_dim=opt.noise_size, label_channel=len(category_names),num_res_blocks=opt.num_res_blocks)
    D_glob = Discriminator(channels=3+len(category_names))
    G_base.cuda()
    D_glob.cuda()  
    D_instance = Discriminator(channels=3+len(category_names))
    D_instance.cuda()

    #Load parameters from pre-trained models
    if opt.pre_trained_model_path != None and opt.pre_trained_model_epoch != None:
        try:
            G_base.load_state_dict(torch.load(opt.pre_trained_model_path + 'G_base_epoch_' + opt.pre_trained_model_epoch))
            D_glob.load_state_dict(torch.load(opt.pre_trained_model_path + 'D_glob_epoch_' + opt.pre_trained_model_epoch))
            D_instance.load_state_dict(torch.load(opt.pre_trained_model_path + 'D_local_epoch_' + opt.pre_trained_model_epoch))    
            print('Parameters are loaded!')
        except:
            print('Error: Pre-trained parameters are not loaded!')
            pass    

    #Define interpolation operation
    up_instance =  nn.Upsample(size=(opt.local_patch_size,opt.local_patch_size),mode='bilinear')
  
    #Define training loss function - binary cross entropy
    BCE_loss = nn.BCELoss()
    
    #Define feature matching loss
    criterionVGG = VGGLoss()
    criterionVGG = criterionVGG.cuda()
    
    #Define optimizer
    G_local_optimizer = optim.Adam(G_base.parameters(), lr=opt.lr, betas=(0.0, 0.9))
    D_local_optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, D_glob.parameters())) +list(filter(lambda p: p.requires_grad, D_instance.parameters())) 
                                   , lr=opt.lr, betas=(0.0,0.9))
    #Deine learning rate scheduler
    scheduler_G = lr_scheduler.StepLR(G_local_optimizer, step_size=opt.optim_step_size, gamma=opt.optim_gamma)
    scheduler_D = lr_scheduler.StepLR(D_local_optimizer, step_size=opt.optim_step_size, gamma=opt.optim_gamma)
    
    
    #----------------------------TRAIN-----------------------------------------
    print('training start!')
    start_time = time.time()
    
    for epoch in range(opt.train_epoch):
        scheduler_G.step()
        scheduler_D.step()
         
        D_local_losses = []
        G_local_losses = []
    
        y_real_ = torch.ones(opt.batch_size)
        y_fake_ = torch.zeros(opt.batch_size)
        y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        epoch_start_time = time.time()
    
        data_iter = iter(train_loader)
        num_iter = 0
        while num_iter < len(train_loader):
            
            j=0
            while j < opt.critic_iter and num_iter < len(train_loader):
                j += 1
                sample_batched = data_iter.next()  
                num_iter += 1            
                x_ = sample_batched['image']
                y_ = sample_batched['seg_mask']
                y_instances = sample_batched['mask_instance']
                bbox = sample_batched['bbox']
                
                mini_batch = x_.size()[0]
                if mini_batch != opt.batch_size:
                    break
                
                #Update discriminators - D 
                #Real examples
                D_glob.zero_grad()
                   
                x_, y_ = Variable(x_.cuda()) , Variable(y_.cuda()) 
                x_d = torch.cat([x_,y_],1)
                
                x_instances = torch.zeros((opt.batch_size,3,128,128))
                x_instances = Variable(x_instances.cuda())
                y_instances = Variable(y_instances.cuda())
                G_instances = torch.zeros((opt.batch_size,3,128,128))
                G_instances = Variable(G_instances.cuda())
                
                #Obtain instances
                for t in range(x_d.size()[0]):
                    x_instance = x_[t,0:3,bbox[0][t]:bbox[1][t],bbox[2][t]:bbox[3][t]] 
                    x_instance = x_instance.contiguous().view(1,x_instance.size()[0],x_instance.size()[1],x_instance.size()[2]) 
                    x_instances[t] = up_instance(x_instance)
                
                
                D_result_instance = D_instance(torch.cat([x_instances,y_instances],1).detach()).squeeze()    
                D_result = D_glob(x_d.detach()).squeeze()
                D_real_loss = BCE_loss(D_result, y_real_) +  BCE_loss(D_result_instance, y_real_)
                D_real_loss.backward()
                
                #Fake examples
                z_ = torch.randn((mini_batch, opt.noise_size))
                z_ = Variable(z_.cuda())
        
                G_result = G_base(z_, y_)
                G_result_d = torch.cat([G_result,y_],1) 
                
                #Obtain fake instances
                for t in range(x_d.size()[0]):
                    G_instance = G_result_d[t,0:3,bbox[0][t]:bbox[1][t],bbox[2][t]:bbox[3][t]] 
                    G_instance = G_instance.contiguous().view(1,G_instance.size()[0],G_instance.size()[1],G_instance.size()[2]) 
                    G_instances[t] = up_instance(G_instance)
                
                
                D_result_instance = D_instance(torch.cat([G_instances,y_instances],1).detach()).squeeze() 
                D_result = D_glob(G_result_d.detach()).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_) +  BCE_loss(D_result_instance, y_fake_)
                D_fake_loss.backward()
                D_local_optimizer.step()
                D_train_loss = D_real_loss + D_fake_loss
                D_local_losses.append(D_train_loss.data[0])
        
            if mini_batch != opt.batch_size:
                break  
            
            #Update generator G
            G_base.zero_grad()   
            
            D_result = D_glob(G_result_d).squeeze() 
            D_result_instance = D_instance(torch.cat([G_instances,y_instances],1)).squeeze() 
            G_train_loss = (1-opt.trade_off_G)*BCE_loss(D_result, y_real_) + opt.trade_off_G*BCE_loss(D_result_instance, y_real_) 
               
            #Feature matching loss between generated image and corresponding ground truth
            FM_loss = criterionVGG(G_result,x_)
               
            total_loss = G_train_loss + opt.lambda_FM*FM_loss 
            total_loss.backward()
            G_local_optimizer.step()
            G_local_losses.append(G_train_loss.data[0])
    
            print('loss_d: %.3f, loss_g: %.3f' % (D_train_loss.data[0],G_train_loss.data[0]))
            if (num_iter % 100) == 0:
                print('%d - %d complete!' % ((epoch+1), num_iter))
                print(result_folder_name)
    
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), opt.train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_local_losses)),
                                                                  torch.mean(torch.FloatTensor(G_local_losses))))

        #Save images
        if epoch == 0:
            for t in range(y_fixed.size()[1]):
                show_result((epoch+1), y_fixed[:,t:t+1,:,:] ,save=True, path=root + result_folder_name+ '/' + model + str(epoch + 1 ) + '_masked.png')
            
        show_result((epoch+1),G_base(z_fixed, y_fixed) ,save=True, path=root + result_folder_name+ '/' + model + str(epoch + 1 ) + '.png')
        
        #Save model params
        if opt.save_models and (epoch>21 and epoch % 10 == 0 ):
            torch.save(G_base.state_dict(), root +model_folder_name+ '/' + model + 'G_base_epoch_'+str(epoch)+'.pth')
            torch.save(D_glob.state_dict(), root + model_folder_name +'/'+ model + 'D_glob_epoch_'+str(epoch)+'.pth')
            torch.save(D_instance.state_dict(), root +model_folder_name+ '/' + model + 'D_local_epoch_'+str(epoch)+'.pth')
              
            
    end_time = time.time()
    total_ptime = end_time - start_time
    print("Training finish!... save training results")
    print('Training time: ' + str(total_ptime))
    
if __name__ == '__main__':
    main()