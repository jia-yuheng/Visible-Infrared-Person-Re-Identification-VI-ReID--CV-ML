from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net, embed_net_mix
from utils_AGW import *
#from utils import *
import torch.nn.functional as F
from loss import *
from tensorboardX import SummaryWriter
from transform import RandomGray, RandomErasing
from ChannelAug import *
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()

def main():

    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
    parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='network baseline:resnet18 or resnet50')
    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--test-only', action='store_true', help='test only')
    parser.add_argument('--model_path', default='save_model/', type=str,
                        help='model save path')
    parser.add_argument('--save_epoch', default=20, type=int,
                        metavar='s', help='save model every 10 epochs')
    parser.add_argument('--log_path', default='log/', type=str,
                        help='log save path')
    parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                        help='log save path')
    parser.add_argument('--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--img_w', default=144, type=int,
                        metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=388, type=int,
                        metavar='imgh', help='img height')
    parser.add_argument('--batch-size', default=8, type=int,
                        metavar='B', help='training batch size')
    parser.add_argument('--test-batch', default=64, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--method', default='base', type=str,
                        metavar='m', help='method type: base or agw')
    parser.add_argument('--margin', default=0.35, type=float,
                        metavar='margin', help='triplet loss margin')
    parser.add_argument('--num_pos', default=4, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial (only for RegDB dataset)')
    parser.add_argument('--seed', default=0, type=int,
                        metavar='t', help='random seed')
    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--mix', default='on', type=str, help='on or off')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    set_seed(args.seed)

    dataset = args.dataset
    if dataset == 'sysu':
        data_path = '/root/autodl-tmp/SYSU-MM01'
        log_path = args.log_path + 'sysu_log/'
        test_mode = [1, 2]  # thermal to visible
    elif dataset == 'regdb':
        data_path = '/root/autodl-tmp/RegDB/'
        log_path = args.log_path + 'regdb_log/'
        test_mode = [1, 2]  # visible to thermal

    checkpoint_path = args.model_path

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.isdir(args.vis_log_path):
        os.makedirs(args.vis_log_path)

    suffix = dataset
    if args.method=='agw':
        suffix = suffix + '_agw_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
    else:
        suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)


    if not args.optim == 'sgd':
        suffix = suffix + '_' + args.optim

    if dataset == 'regdb':
        suffix = suffix + '_trial_{}'.format(args.trial)

    sys.stdout = Logger(log_path + suffix + '_os.txt')

    vis_log_dir = args.vis_log_path + suffix + '/'

    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    writer = SummaryWriter(vis_log_dir)
    print("==========\nArgs:{}\n==========".format(args))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0

    print('==> Loading data..')
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train_gobal = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ChannelAdapGray(probability =0.5),
        RandomErasing(),
        normalize,
    ])
    transform_train_part = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((194, 144)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ChannelAdapGray(probability =0.5),
        RandomErasing(),
        normalize,
    ])

    transform_test_gobal = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test_part = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((194, 144)),
        transforms.ToTensor(),
        normalize,
    ])
    #cyclegan = GeneratorResNet((3,288,144),9)
    #cyclegan.load_state_dict(torch.load('./G_BA_100.pth'))
    #cyclegan.to(device)
    end = time.time()
    if dataset == 'sysu':
        # training set
        trainset = SYSUData(data_path, transform1=transform_train_gobal, transform2=transform_train_part)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    elif dataset == 'regdb':
        # training set
        trainset = RegDBData(data_path, args.trial, transform1=transform_train_gobal, transform2=transform_train_part)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='visible')

    gallset = TestData(gall_img, gall_label, transform1=transform_test_gobal, transform2=transform_test_part, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform1=transform_test_gobal, transform2=transform_test_part, img_size=(args.img_w, args.img_h))

    # testing data loader
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    n_class = len(np.unique(trainset.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    print('Dataset {} statistics:'.format(dataset))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
    print('  ------------------------------')
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    print('==> Building model..')
    if args.method =='agw':
        net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
    if args.method =='base' and args.mix == 'on':
        net = embed_net_mix(n_class, non_local='off', arch=args.arch)

    elif args.method =='base' and args.mix == 'off':
        net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)
    else:
        net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
    net.to(device)


    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # define loss function
    criterion_id_gobal = nn.CrossEntropyLoss()
    criterion_ensemble = CrossEntropyLabelSmooth(n_class)
    criterion_id_part = nn.CrossEntropyLoss()
    if args.method == 'agw':
        criterion_tri = TripletLoss_WRT()
    else:
        loader_batch = args.batch_size * args.num_pos
        criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
        #criterion_tri= CenterTripletLoss(batch_size=loader_batch, margin=args.margin)
        #criterion_tri = TripletLoss_WRT()

    criterion_id_gobal.to(device)
    criterion_id_part.to(device)
    criterion_tri.to(device)
    criterion_mmd = MMD_Loss().to(device)
    print(args.mix)

    if args.optim == 'sgd' and args.mix == 'on':
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.bottleneck_part.parameters())) \
                         + list(map(id, net.bottleneck_mix.parameters())) \
                         + list(map(id, net.classifier_gobal.parameters())) \
                         + list(map(id, net.classifier_part.parameters())) \
                         + list(map(id, net.classifier_mix.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.bottleneck_part.parameters(), 'lr': args.lr},
            {'params': net.bottleneck_mix.parameters(), 'lr': args.lr},
            {'params': net.classifier_gobal.parameters(), 'lr': args.lr},
            {'params': net.classifier_part.parameters(), 'lr': args.lr},
            {'params': net.classifier_mix.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)

    if args.optim == 'sgd' and args.mix == 'off':
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if epoch < 10:
            lr = args.lr * (epoch + 1) / 10
        elif epoch >= 10 and epoch < 20:
            lr = args.lr
        elif epoch >= 20 and epoch < 30:
            lr = args.lr * 0.1
        elif epoch >= 30 and epoch < 50:
            lr = 0.01
        elif epoch >= 50:
            lr = args.lr * 0.01


        optimizer.param_groups[0]['lr'] = 0.1 * lr
        for i in range(len(optimizer.param_groups) - 1):
            optimizer.param_groups[i + 1]['lr'] = lr

        return lr


    def train(epoch, part='off'):

        current_lr = adjust_learning_rate(optimizer, epoch)
        train_loss = AverageMeter()
        id_loss = AverageMeter()
        tri_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        correct_gobal = 0
        correct_part = 0
        correct_mix = 0
        total = 0

        # switch to train mode
        net.train()
        end = time.time()

        for batch_idx, (input1_gobal, input2_gobal, input1_part, input2_part, label1, label2) in enumerate(trainloader):
            #print(input1_gobal.shape)
            if args.mix == 'on':
                #print(batch_idx)

                labels = torch.cat((label1, label2), 0).long()
                labels = labels.cuda()

                input1_gobal = Variable(input1_gobal.cuda())
                input2_gobal = Variable(input2_gobal.cuda())

                input1_part = Variable(input1_part.cuda())
                input2_part = Variable(input2_part.cuda())

                labels = Variable(labels.cuda())
                data_time.update(time.time() - end)

                feat_gobal, feat_part, feat_intra_mix, feat_inter_mix, out0_gobal, out0_part, out0_intra_mix, out0_inter_mix = net(input1_gobal, input2_gobal, input1_part, input2_part)
                feat_rgb_mix, feat_ir_mix = torch.split(feat_intra_mix, [32,32], dim=0)
                feat_rgb_gobal, feat_ir_gobal = torch.split(feat_gobal, [32,32], dim=0)
                feat_rgb_inter_mix, feat_ir_inter_mix = torch.split(feat_inter_mix, [32,32], dim=0)
                loss_mmd = criterion_mmd(feat_rgb_mix, feat_ir_mix) + criterion_mmd(feat_rgb_inter_mix, feat_ir_inter_mix)

                loss_id_gobal = criterion_id_gobal(out0_gobal, labels)
                loss_id_part = criterion_id_part(out0_part, labels)
                loss_tri_gobal, batch_acc_gobal = criterion_tri(feat_gobal, labels)
                correct_gobal += (batch_acc_gobal / 2)
                _, predicted_gobal = out0_gobal.max(1)
                correct_gobal += (predicted_gobal.eq(labels).sum().item() / 2)
                loss_gobal = loss_id_gobal + loss_tri_gobal

                # feat_part, out0_part, = net_part(input1_part, input2_part)
                # # print(out0.shape)
                loss_tri_part, batch_acc_part = criterion_tri(feat_part, labels)
                correct_part += (batch_acc_part / 2)
                _, predicted_part = out0_part.max(1)
                correct_part += (predicted_part.eq(labels).sum().item() / 2)
                loss_part = loss_id_part + loss_tri_part

                loss_id_mix = criterion_id_gobal(out0_intra_mix, labels) + criterion_id_gobal(out0_inter_mix, labels)
                #loss_id_mix = criterion_ensemble(out0_mix, labels)
                loss_tri_mix, batch_acc_mix = criterion_tri(feat_intra_mix,labels)
                correct_mix += (batch_acc_mix / 2)
                _, predicted_mix = out0_intra_mix.max(1)
                correct_mix += (predicted_mix.eq(labels).sum().item() / 2)
                loss_mix = loss_id_mix + loss_tri_mix + criterion_tri(feat_inter_mix,labels)[0]

                loss_id = (loss_id_gobal + loss_id_part + loss_id_mix)/3
                loss_tri = (loss_tri_gobal + loss_tri_part +loss_tri_mix)/3
                loss = loss_gobal + loss_part + loss_mix + loss_mmd
                correct = (correct_gobal + correct_mix + correct_part)/3
            else:

                labels = torch.cat((label1, label2), 0).long()
                labels = labels.cuda()

                input1 = Variable(input1_gobal.cuda())
                input2 = Variable(input2_gobal.cuda())

                labels = Variable(labels.cuda())
                data_time.update(time.time() - end)

                feat, out0, = net(input1, input2)
                loss_id = criterion_id_gobal(out0, labels)
                loss_tri, batch_acc = criterion_tri(feat, labels)
                correct += (batch_acc / 2)
                _, predicted = out0.max(1)

                correct += (predicted.eq(labels).sum().item() / 2)

                loss = loss_id + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update P
            train_loss.update(loss.item(), 2 * input1_gobal.size(0))
            id_loss.update(loss_id.item(), 2 * input1_gobal.size(0))
            tri_loss.update(loss_tri.item(), 2 * input1_gobal.size(0))
            total += labels.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 50 == 0:
                print('Epoch: [{}][{}/{}] '
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'lr:{:.3f} '
                      'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                      'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                      'Accu: {:.2f}'.format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total, batch_time=batch_time,
                    train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

        writer.add_scalar('total_loss', train_loss.avg, epoch)
        writer.add_scalar('id_loss', id_loss.avg, epoch)
        writer.add_scalar('tri_loss', tri_loss.avg, epoch)
        writer.add_scalar('lr', current_lr, epoch)


    def test(epoch):
        # switch to evaluation mode
        if args.mix == 'on':
            net.eval()
            print('Extracting Gallery Feature...')
            start = time.time()
            ptr = 0
            gall_feat = np.zeros((ngall, 4096))
            gall_feat_att = np.zeros((ngall, 4096))
            with torch.no_grad():
                for batch_idx, (input_gobal, input_part, label) in enumerate(gall_loader):
                    batch_num = input_gobal.size(0)
                    input_gobal = Variable(input_gobal.cuda())
                    input_part = Variable(input_part.cuda())
                    feat_gobal, feat_part, feat_mix, feat_att_gobal, feat_att_part, feat_att_mix= net(input_gobal, input_gobal, input_part, input_part, test_mode[0])
                    gall_feat[ptr:ptr + batch_num, :] = feat_mix.detach().cpu().numpy()
                    gall_feat_att[ptr:ptr + batch_num, :] = feat_att_mix.detach().cpu().numpy()
                    ptr = ptr + batch_num
            print('Extracting Time:\t {:.3f}'.format(time.time() - start))

            # switch to evaluation
            net.eval()
            print('Extracting Query Feature...')
            start = time.time()
            ptr = 0
            query_feat = np.zeros((nquery, 4096))
            query_feat_att = np.zeros((nquery, 4096))
            with torch.no_grad():
                for batch_idx, (input_gobal, input_part, label) in enumerate(query_loader):
                    batch_num = input_gobal.size(0)
                    input_gobal = Variable(input_gobal.cuda())
                    input_part = Variable(input_part.cuda())
                    feat_gobal, feat_part, feat_mix, feat_att_gobal, feat_att_part, feat_att_mix = net(input_gobal, input_gobal, input_part, input_part, test_mode[1])
                    query_feat[ptr:ptr + batch_num, :] = feat_mix.detach().cpu().numpy()
                    query_feat_att[ptr:ptr + batch_num, :] = feat_att_mix.detach().cpu().numpy()
                    ptr = ptr + batch_num
            print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        else:
            net.eval()
            print('Extracting Gallery Feature...')
            start = time.time()
            ptr = 0
            gall_feat = np.zeros((ngall, 2048))
            gall_feat_att = np.zeros((ngall, 2048))
            with torch.no_grad():
                for batch_idx, (input_gobal, input_part, label) in enumerate(gall_loader):
                    batch_num = input_gobal.size(0)
                    input_gobal = Variable(input_gobal.cuda())
                    input_part = Variable(input_part.cuda())
                    feat_gobal, feat_att_gobal = net(input_gobal, input_gobal, test_mode[0])
                    gall_feat[ptr:ptr + batch_num, :] = feat_gobal.detach().cpu().numpy()
                    gall_feat_att[ptr:ptr + batch_num, :] = feat_att_gobal.detach().cpu().numpy()
                    ptr = ptr + batch_num
            print('Extracting Time:\t {:.3f}'.format(time.time() - start))

            # switch to evaluation
            net.eval()
            print('Extracting Query Feature...')
            start = time.time()
            ptr = 0
            query_feat = np.zeros((nquery, 2048))
            query_feat_att = np.zeros((nquery, 2048))
            with torch.no_grad():
                for batch_idx, (input_gobal, input_part, label) in enumerate(query_loader):
                    batch_num = input_gobal.size(0)
                    input_gobal = Variable(input_gobal.cuda())
                    input_part = Variable(input_part.cuda())
                    feat_gobal, feat_att_gobal = net(input_gobal, input_gobal, test_mode[1])
                    query_feat[ptr:ptr + batch_num, :] = feat_gobal.detach().cpu().numpy()
                    query_feat_att[ptr:ptr + batch_num, :] = feat_att_gobal.detach().cpu().numpy()
                    ptr = ptr + batch_num
            print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        start = time.time()
        # compute the similarity
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

        # evaluation
        if dataset == 'regdb':
            cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
            cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
        elif dataset == 'sysu':
            cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

        writer.add_scalar('rank1', cmc[0], epoch)
        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('mINP', mINP, epoch)
        writer.add_scalar('rank1_att', cmc_att[0], epoch)
        writer.add_scalar('mAP_att', mAP_att, epoch)
        writer.add_scalar('mINP_att', mINP_att, epoch)
        return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


    # training
    print('==> Start Training...')
    for epoch in range(start_epoch, 81 - start_epoch):

        print('==> Preparing Data Loader...')
        # identity sampler
        sampler = IdentitySampler(trainset.train_color_label, \
                                  trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                                  epoch)

        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # thermal index
        print(epoch)
        print(trainset.cIndex)
        print(trainset.tIndex)

        loader_batch = args.batch_size * args.num_pos

        trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                      sampler=sampler, num_workers=args.workers, drop_last=True)

        # training
        train(epoch, n_class)

        if epoch > 0 and epoch % 2 == 0:
            print('Test Epoch: {}'.format(epoch))

            # testing
            cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
            # save model
            if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
                best_acc = cmc_att[0]
                best_epoch = epoch
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc_att,
                    'mAP': mAP_att,
                    'mINP': mINP_att,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_best.t')

            # save model
            # if epoch > 10 and epoch % args.save_epoch == 0:
            #     state = {
            #         'net': net.state_dict(),
            #         'cmc': cmc,
            #         'mAP': mAP,
            #         'epoch': epoch,
            #     }
            #     torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

            print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
            print('Best Epoch [{}]'.format(best_epoch))



if __name__ == '__main__':
    main()