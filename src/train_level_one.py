import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from data_read import *
from net import *
import numpy as np
import os
import settings
from train_utils import initial_motion, gt_motion_rs, calculate_feature_map, compute_ssim, gt_motion_rs_random_noisy

os.environ['CUDA_VISIBLE_DEVICES']='3'

rho = settings.rho
mu = settings.mu
lambda_loss = settings.lambda_loss
batch_size = settings.batch_size
lr = settings.lr

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

save_path = f'{settings.path}/training'+'/level_one_un_4_1_con/'
path_split = save_path.split('/')
make_path = '../'
for i in range(1, len(path_split)):
    make_path = make_path + path_split[i] + '/'
    if not(os.path.exists(make_path)):
        os.mkdir(make_path)

level_one_input = ResNet_first_input()
level_one_template = ResNet_first_template()
corr_uncertainty = corr_uncertainty_decoder4()

def local_cost_volume(c1, warp, search_range):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    # cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

    return cost_vol
####


initial_matrix = initial_motion(batch_size)
LK_layer_one = Lucas_Kanade_layer(batch_size=batch_size,height_template=128,width_template=128,num_channels=1)


dfhistory = pd.DataFrame(columns=['epoch', 'convex_loss', 'equ1_loss', 'equ2_loss', 'equ3_loss', 'ssim_loss', 'total_loss'])

for current_epoch in range(settings.train_stage1):
    if settings.dataset_name=='MSCOCO':
        data_loader_caller=data_loader_MSCOCO('train')
    if settings.dataset_name=='GoogleMap':
        data_loader_caller=data_loader_GoogleMap('train')
    if settings.dataset_name=='GoogleEarth':
        data_loader_caller=data_loader_GoogleEarth('train')
    if settings.dataset_name=='DPDN':
        data_loader_caller=data_loader_DPDN('train')
        
    optimizer = tf.keras.optimizers.Adam(lr=lr,beta_1=0.9)
    print("Starting epoch " + str(current_epoch+1))
    print("Learning rate is " + str(lr)) 

    loss_total = 0.0
    equ1_loss_total = 0.0
    equ2_loss_total = 0.0
    equ3_loss_total = 0.0
    ssim_loss_total = 0.0
    convex_loss_total = 0.0
    ssim_loss_total = 0.0
    count = 0
    for iters in range(100000000):
        input_img,u_list,v_list,template_img=data_loader_caller.data_read_batch(batch_size=batch_size)
        count += 1
        
        if len(np.shape(input_img))<2:
          level_one_input.save_weights(save_path +'epoch_'+str(1+current_epoch)+"input_full")
          level_one_template.save_weights(save_path +'epoch_'+str(1+current_epoch)+"template_full")
          corr_uncertainty.save_weights(save_path + 'epoch_' + str(1 + current_epoch) + "un")
          break

        gt_matrix_one=gt_motion_rs(u_list,v_list,batch_size=batch_size)
        with tf.GradientTape() as tape:
            input_feature = level_one_input.call(input_img)
            template_feature = level_one_template.call(template_img)

            input_feature_map_one = calculate_feature_map(input_feature)
            template_feature_map_one = calculate_feature_map(template_feature)

            input_warped_to_template = LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one)

            ssim_middle_one = tf.reduce_mean(compute_ssim(template_feature_map_one,input_warped_to_template))

            corr_attention = local_cost_volume(template_feature_map_one, input_warped_to_template, 4)
            log_var = corr_uncertainty.call(corr_attention)
            B, H, W, C = input_warped_to_template.shape
            up_log_var = tf.image.resize(log_var, size=(H, W), method=tf.image.ResizeMethod.BILINEAR)
            ssim_middle = ssim_middle_one + tf.reduce_mean(tf.multiply(tf.exp(-up_log_var),compute_ssim(template_feature_map_one,input_warped_to_template)) + up_log_var)  #



            for nn in range(settings.sample_noise):
                lambda_one=(np.random.rand(8)-0.5)/6


                for mm in range(len(lambda_one)):
                  if lambda_one[mm]>0 and lambda_one[mm]<0.02:
                    lambda_one[mm]=0.02
                  if lambda_one[mm]<0 and lambda_one[mm]>-0.02:
                    lambda_one[mm]=-0.02
          
                noise_one = gt_motion_rs_random_noisy(u_list,v_list,batch_size=batch_size,lambda_noisy=lambda_one)

                lambda_noisy_labels_one_plus = LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one + rho * noise_one)
                h_lambda_noisy_labels_one_plus = tf.reduce_mean(compute_ssim(template_feature_map_one, lambda_noisy_labels_one_plus))

                lambda_noisy_labels_one_mins = LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one- rho * noise_one)
                h_lambda_noisy_labels_one_mins = tf.reduce_mean(compute_ssim(template_feature_map_one,lambda_noisy_labels_one_mins))

                noisy_labels_one_plus = LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one+noise_one)
                h_noisy_labels_one_plus = tf.reduce_mean(compute_ssim(template_feature_map_one,noisy_labels_one_plus))

                noisy_labels_one_mins = LK_layer_one.projective_inverse_warp(input_feature_map_one, gt_matrix_one-noise_one)
                h_noisy_labels_one_mins = tf.reduce_mean(compute_ssim(template_feature_map_one,noisy_labels_one_mins))
                
                if settings.option == 'option1':
                    # equ1
                    equ1_plus= 0
                    equ1_mins= 0
                elif settings.option == 'option2':
                    # equ1
                    equ1_plus= tf.math.maximum(ssim_middle_one - h_lambda_noisy_labels_one_plus, 0)
                    equ1_mins= tf.math.maximum(ssim_middle_one - h_lambda_noisy_labels_one_mins, 0)
                # equ2
                equ2_plus = tf.math.maximum(ssim_middle_one + mu * np.sum(lambda_one**2) / 2 - h_noisy_labels_one_plus,0)
                equ2_mins = tf.math.maximum(ssim_middle_one + mu * np.sum(lambda_one**2) / 2 - h_noisy_labels_one_mins,0)
                # equ3
                equ3_plus = tf.math.maximum(h_lambda_noisy_labels_one_plus - (1 - rho) * ssim_middle_one - rho * h_noisy_labels_one_plus + mu * rho * (1 - rho) * np.sum(lambda_one**2) / 2 ,0)
                equ3_mins = tf.math.maximum(h_lambda_noisy_labels_one_mins - (1 - rho) * ssim_middle_one - rho * h_noisy_labels_one_mins + mu * rho * (1 - rho) * np.sum(lambda_one**2) / 2 ,0)
         
                if nn==0:
                    convex_loss = equ1_mins + equ1_plus + equ2_mins + equ2_plus + equ3_mins + equ3_plus
                    equ1_loss = equ1_mins + equ1_plus
                    equ2_loss = equ2_mins + equ2_plus
                    equ3_loss = equ3_mins + equ3_plus
                else:
                    convex_loss = convex_loss + equ1_mins + equ1_plus + equ2_mins + equ2_plus + equ3_mins + equ3_plus
                    equ1_loss = equ1_loss + equ1_mins + equ1_plus
                    equ2_loss = equ2_loss + equ2_mins + equ2_plus
                    equ3_loss = equ3_loss + equ3_mins + equ3_plus

            total_loss = ssim_middle + lambda_loss * convex_loss

            convex_loss_total += convex_loss
            equ1_loss_total += equ1_loss
            equ2_loss_total += equ2_loss
            equ3_loss_total += equ3_loss
            ssim_loss_total += ssim_middle
            loss_total += total_loss



        all_parameters = level_one_template.trainable_variables+level_one_input.trainable_variables + corr_uncertainty.trainable_variables
           
        grads = tape.gradient(total_loss, all_parameters)
        grads = [tf.clip_by_value(i,-0.1,0.1) for i in grads]
        optimizer.apply_gradients(zip(grads, all_parameters))

        input_img = None
        u_list = None
        v_list = None
        template_img = None

        input_feature_map = None
        template_feature_map = None
        input_warped_to_template=None
        input_warped_to_template_left_1=None
        input_warped_to_template_left_2=None
        input_warped_to_template_right_1=None
        input_warped_to_template_right_2=None


    print('total_loss', loss_total)
    info = (int(current_epoch+1), float(convex_loss_total) / count, float(equ1_loss_total) / count, float(equ2_loss_total) / count, float(equ3_loss_total) / count, float(ssim_loss_total) / count, float(loss_total) / count)
    dfhistory.loc[current_epoch] = info
    dfhistory.to_csv(f'{settings.path}/training/dfhistory_level_un1_4_1.csv', index=False)
 
    loss_total=0.0
    convex_loss_total=0.0
    equ1_loss_total=0.0
    equ2_loss_total=0.0
    equ3_loss_total=0.0
    ssim_loss_total=0.0

    level_one_input.save_weights(save_path +'epoch_'+str(1+current_epoch)+"input_"+str(iters))
    level_one_template.save_weights(save_path +'epoch_'+str(1+current_epoch)+"template_"+str(iters))
    corr_uncertainty.save_weights(save_path + 'epoch_' + str(1 + current_epoch) + "un"+str(iters))


           
        

               




