import torch
import torch.nn as nn
from torch import optim
import visdom
from torch.utils.data import DataLoader
from MobileNet.mobilenet_v1 import MobileNet
from MobileNet.iris_csv import Iris

batch_size=16
base_learning_rate=1e-4

epoches=10
torch.manual_seed(1234)
vis=visdom.Visdom()
train_db=Iris('imagedourcepath',64,128,'train')
validation_db=Iris('imagedourcepath',64,128,'validation')
test_db=Iris('imagedourcepath',64,128,'test')

train_loader=DataLoader(train_db,batch_size=batch_size,shuffle=True,num_workers=4)
validation_loader=DataLoader(validation_db,batch_size=batch_size,num_workers=2)
test_loader=DataLoader(test_db,batch_size=batch_size,num_workers=2)
def evaluate(model,loader):
    correct=0
    total_num=len(loader.dataset)
    for x,y in loader:
        # x,y=x.to(device),y.to(device)
        with torch.no_grad():
            logits=model(x)
            pred=logits.argmax(dim=1)
        correct+=torch.eq(pred,y).sum().float().item()
    return correct/total_num
def adapt_weights(pthfile,module):
    module_dict=module.state_dict()
    pretrained_dict=torch.load(pthfile)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in module_dict}
    module_dict.update(pretrained_dict)
    module.load_state_dict(module_dict)

def main():
    mod=MobileNet(35)
    mod_dict = mod.state_dict()
    nn.init.kaiming_normal_(mod.upchannel.weight, nonlinearity='relu')
    nn.init.constant_(mod.upchannel.bias,0.1)
    pretrained_dict = torch.load('root/tf_to_torch.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in mod_dict}
    mod_dict.update(pretrained_dict)
    mod.load_state_dict(mod_dict)
    freeze_list=list(mod.state_dict().keys())[0:-2]
    # print(freeze_list)
    for name,param in mod.named_parameters():
         if name in freeze_list:
             param.requires_grad=False
         if param.requires_grad:
             print(name)
    optimizer=optim.SGD(filter(lambda p: p.requires_grad, mod.parameters()),lr=base_learning_rate)
    fun_loss = nn.CrossEntropyLoss()
    vis.line([0.], [-1], win='train_loss', opts=dict(title='train_loss'))
    vis.line([0.], [-1], win='validation_acc', opts=dict(title='validation_acc'))
    global_step = 0
    best_epoch, best_acc = 0, 0
    for epoch in range(10):
        for step, (x, y) in enumerate(train_loader):
            logits = mod(x)
            # print(logits.shape)
            loss = fun_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            vis.line([loss.item()], [global_step], win='train_loss', update='append')
            global_step += 1


        if epoch%1==0:
            val_acc = evaluate(mod, validation_loader)
            if  val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(mod.state_dict(), 'best.pth')
                vis.line([val_acc], [global_step], win='validation_acc', update='append')

    print('best acc', best_acc, 'best epoch', best_epoch)

if __name__ == '__main__':
    main()