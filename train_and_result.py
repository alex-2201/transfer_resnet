from train import train_model
from load_pretrain import model_ft, criterion, optimizer_ft, exp_lr_scheduler
from visual_result import visualize_model

if __name__ == "__main__":
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=25)

    visualize_model(model_ft)
