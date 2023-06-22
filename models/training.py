from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from glob import glob
import numpy as np
from models.generation import Generator
from torch.utils.tensorboard import SummaryWriter
import trimesh
from data_processing import utils

from tqdm import tqdm
import time

class Trainer(object):

    def __init__(self, model, device, train_dataset, val_dataset, exp_name, optimizer='Adam'):
        self.model = model.to(device)
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = '{}/{}/'.format( train_dataset.cfg['experiment_prefix'], exp_name)
        print(self.exp_path)
        self.checkpoint_path = self.exp_path + 'checkpoints/'
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.val_min = None

        self.writer = SummaryWriter(log_dir=self.exp_path + 'summary'.format(exp_name))
        self.gen = Generator(model, self.val_dataset.cfg)
        
        self.val_loader = self.val_dataset.get_loader(shuffle=False)
        self.train_loader = self.train_dataset.get_loader()

    def train_step(self, batch, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        loss_shape, loss_color, loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss_shape, loss_color, loss.detach()

    def compute_val_loss(self, epoch):
        self.model.eval()

        sum_val_loss = 0
        sum_val_loss_color = 0
        sum_val_loss_shape = 0

        it = 0
        start_time = time.time()
        for val_batch in self.val_loader:
            with torch.no_grad():
                loss_shape, loss_color, loss = self.compute_loss(val_batch)
            sum_val_loss += loss.item()
            sum_val_loss_color += loss_color
            sum_val_loss_shape += loss_shape

            if it == 0:
                print("Adding meshes..")
                self.add_meshes(val_batch, epoch)

            it += 1
            print("Epoch %d validation %d/%d" % (epoch, it, len(self.val_loader)))

        print("Time elapsed for validation loss: %f" % ((time.time() - start_time)/60.))
        num_batches = len(self.val_loader)
        return sum_val_loss_shape/num_batches, sum_val_loss_color/num_batches, sum_val_loss / num_batches

    def compute_loss(self,batch):
        loss_shape = 0.01*self.compute_shape_loss(batch)
        loss_color = self.compute_color_loss(batch)
        print(loss_shape, loss_color)
        loss = loss_shape + loss_color
        return float(loss_shape.detach().item()), float(loss_color.detach().item()), loss

    def compute_shape_loss(self,batch):
        device = self.device
        p = batch.get('grid_coords').to(device)
        gt_occ = batch.get('occ').to(device)
        inputs_shape = batch.get('inputs_shape').to(device) # (B, 1, W, H, D)
        pred_shape = self.model.forward_shape(p, inputs_shape)

        loss_shape = F.binary_cross_entropy_with_logits(pred_shape, gt_occ, reduction='none')
        loss_shape = loss_shape.sum(-1).mean()
        return loss_shape


    def compute_color_loss(self,batch):
        device = self.device
        p_surface = batch.get('grid_coords_surface').to(device)
        gt_rgb = batch.get('rgb').to(device)
        inputs = batch.get('inputs').to(device) # (B, C, W, H, D)
        inputs_shape = batch.get('inputs_shape').to(device) # (B, 1, W, H, D)

        pred_rgb = self.model.forward_color(p_surface, inputs, inputs_shape)

        loss_color = torch.nn.L1Loss(reduction='none')(pred_rgb, gt_rgb)
        loss_color = loss_color.sum(-1).mean() # loss_i summed 3 rgb channels for all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)
        return loss_color


    def train_model(self, epochs):
        loss = 0
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            sum_loss = 0
            sum_loss_shape = 0
            sum_loss_color = 0
            print('Start epoch {}'.format(epoch))
            

            if epoch % 1 == 0:
                self.model.eval()
                self.save_checkpoint(epoch)

                val_loss_shape, val_loss_color, val_loss = self.compute_val_loss(epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Loss_shape/val", val_loss_shape, epoch)
                self.writer.add_scalar("Loss_color/val", val_loss_color, epoch)

                if self.val_min is None:
                    self.val_min = val_loss
                if val_loss < self.val_min:
                    self.val_min = val_loss
                    np.save(self.exp_path + 'val_min',[epoch,val_loss])

            it = 0 
            num_batches = len(self.train_loader)
            for batch in self.train_loader:
                loss_shape, loss_color, loss = self.train_step(batch, epoch)
                print("Current loss: {} Batch: {}/{} Epoch: {}/{}".format(loss, it+1, num_batches, epoch+1, epochs))
                sum_loss += loss.item()
                sum_loss_shape += loss_shape
                sum_loss_color += loss_color
                it += 1
            
            self.writer.add_scalar("Loss/train", sum_loss / len(self.train_loader), epoch)
            self.writer.add_scalar("Loss_shape/train", sum_loss_shape / len(self.train_loader), epoch)
            self.writer.add_scalar("Loss_color/train", sum_loss_color / len(self.train_loader), epoch)

    def add_meshes(self, batch, epoch, n_meshes=3):
        self.gen.model = self.model
        n = min(len(batch['path']), n_meshes)
        for j in range(n):
            in_mesh = trimesh.load(batch['path'][j])
            out_mesh = self.gen.generate_mesh(batch['inputs'][[j]], batch['inputs_shape'][[j]])
            gt_mesh = trimesh.load(batch['gt_mesh_path'][j])
            self.writer.add_mesh('Meshes/input_{}'.format(j), vertices=np.expand_dims(in_mesh.vertices,0), faces=np.expand_dims(in_mesh.faces,0), colors=np.expand_dims(in_mesh.visual.to_color().vertex_colors[:,:3], 0))
            self.writer.add_mesh('Meshes/output_{}'.format(j), vertices=np.expand_dims(out_mesh.vertices,0), faces=np.expand_dims(out_mesh.faces,0), colors=np.expand_dims(out_mesh.visual.vertex_colors[:,:3], 0), global_step=epoch)
            self.writer.add_mesh('Meshes/ground_truth_{}'.format(j), vertices=np.expand_dims(gt_mesh.vertices,0), faces=np.expand_dims(gt_mesh.faces,0), colors=np.expand_dims(gt_mesh.visual.to_color().vertex_colors[:, :3], 0))

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({'epoch':epoch,'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch