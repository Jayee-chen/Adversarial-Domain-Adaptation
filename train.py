import time
import torch
import os
import math
import ipdb
import torch.nn.functional as F

def train(source_train_loader, source_train_loader_batch, target_train_loader, target_train_loader_batch, model, criterion_classifier_source, criterion_classifier_target, criterion_em_target, criterion, optimizer, epoch, epoch_count_dataset, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_classifier = AverageMeter()
    losses_G = AverageMeter()
    top1_source = AverageMeter()
    top1_target = AverageMeter()
    model.train()
    new_epoch_flag = False
    end = time.time()
    try:
        (input_source, target_source) = source_train_loader_batch.__next__()[1]
    except StopIteration:
        if epoch_count_dataset == 'source':
            epoch = epoch + 1
            new_epoch_flag = True
        source_train_loader_batch = enumerate(source_train_loader)
        (input_source, target_source) = source_train_loader_batch.__next__()[1]

    try:
        (input_target, _) = target_train_loader_batch.__next__()[1]
    except StopIteration:
        if epoch_count_dataset == 'target':
            epoch = epoch + 1
            new_epoch_flag = True
        target_train_loader_batch = enumerate(target_train_loader)
        (input_target, _) = target_train_loader_batch.__next__()[1]
    data_time.update(time.time() - end)

'''cst中下半部分目标分类器的标签'''
target_source_temp = target_source + args.num_classes
target_source_temp = target_source_temp.cuda(async=True)
target_source_temp_var = torch.autograd.Variable(target_source_temp) #### labels for target classifier

'''cst中上半部分源分类器的标签'''
target_source = target_source.cuda(async=True)
input_source_var = torch.autograd.Variable(input_source)
target_source_var = torch.autograd.Variable(target_source) ######## labels for source classifier.
############################################ for source samples
output_source = model(input_source_var) #维度（batch_size,num_classes*2）

'''loss_task_s_Cs将前args.num_classes个类别的预测与源标签进行交叉熵计算；
    loss_task_s_Ct将后args.num_classes个类别的预测与源标签进行交叉熵计算'''

loss_task_s_Cs = criterion(output_source[:,:args.num_classes], target_source_var)
loss_task_s_Ct = criterion(output_source[:,args.num_classes:], target_source_var)

loss_domain_st_Cst_part1 = criterion_classifier_source(output_source)
loss_category_st_G = 0.5 * criterion(output_source, target_source_var) + 0.5 * criterion(output_source, target_source_temp_var)


input_target_var = torch.autograd.Variable(input_target)
output_target = model(input_target_var)
loss_domain_st_Cst_part2 = criterion_classifier_target(output_target)
loss_domain_st_G = 0.5 * criterion_classifier_target(output_target) + 0.5 * criterion_classifier_source(output_target)
loss_target_em = criterion_em_target(output_target)

lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1
if args.flag == 'no_em':
    loss_classifier = loss_task_s_Cs + loss_task_s_Ct + loss_domain_st_Cst_part1 + loss_domain_st_Cst_part2       ### used to classifier
    loss_G = loss_category_st_G + lam * loss_domain_st_G   ### used to feature extractor

elif args.flag == 'symnet':    #
    loss_classifier = loss_task_s_Cs + loss_task_s_Ct + loss_domain_st_Cst_part1 + loss_domain_st_Cst_part2   ### used to classifier
    loss_G = loss_category_st_G + lam * (loss_domain_st_G + loss_target_em)  ### used to feature extractor

else:
    raise ValueError('unrecognized flag:', args.flag)

# mesure accuracy and record loss
prec1_source, _ = accuracy(output_source.data[:, :args.num_classes], target_source, topk=(1,5))
prec1_target, _ = accuracy(output_source.data[:, args.num_classes:], target_source, topk=(1,5))
losses_classifier.update(loss_classifier.data[0], input_source.size(0))
losses_G.update(loss_G.data[0], input_source.size(0))
top1_source.update(prec1_source[0], input_source.size(0))
top1_target.update(prec1_target[0], input_source.size(0))




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
'''获取output中每个样本的前maxk个预测结果，并将结果按大小排序
返回两个张量，第一个是每个样本前k个预测结果的概率值，第二个是其索引'''
_, pred = output.topk(maxk, 1, True, True)
pred = pred.t()
correct = pred.eq(target.view(1, -1).expand_as(pred))
'''将前k个预测结果的准确率通过view（-1）方法将数据展平成一维张量，然后进行求和，得到正确的预测数量。最后返回不同topk值下的准确率列表'''
res = []
for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
return res