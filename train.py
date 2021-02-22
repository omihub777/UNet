import comet_ml
import argparse
from utils import *
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True)
parser.add_argument("--model-name", default="unet")
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--eval-batch-size",default=32, type=int)
parser.add_argument("--max-epochs", default=200, type=int)
parser.add_argument("--optimizer", default="adam")
parser.add_argument("--learning-rate", default=1e-3, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--in-c", default=3, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--size", default=224, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

args.out_c = 1
torch.backends.cudnn.benchmark=True
torch.manual_seed(args.seed)

if args.dry_run:
    print("=====DRY RUN=====")
    args.batch_size = 2
    args.eval_batch_size = 2
    args.max_epochs = 1


scaler=torch.cuda.amp.GradScaler()
class Trainer:
    def __init__(self, args, logger):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.args = args  
        self.logger = logger
        self.model = get_model(args).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = get_optimizer(args, self.model)

    def fit(self, train_dl, test_dl):
        step = 0
        for epoch in range(1, self.args.max_epochs+1):
            epoch_loss, num_imgs = 0., 0.
            self.model.train()
            for image, target in train_dl:
                step+=1
                num_imgs += image.size(0)
                loss = self.trian_step(self.model, image, target, step)
                epoch_loss += loss
            epoch_loss /= num_imgs
            logger.log_metric("loss", epoch_loss, step=step)
            self.model.eval()
            for i, image in enumerate(test_dl):
                if i>=1:
                    break
                self.test_step(self.model, image, step)
            
            if epoch%20 == 0:
                filename=f"weights/{self.args.model_name}_ep{epoch}.pth"
                torch.save(self.model.state_dict(), f=filename)
                self.logger.log_asset(filename, file_name=f"{self.args.model_name}_ep{epoch}.pth")

    def trian_step(self, model, image, target, step):
        self.optimizer.zero_grad()
        image, target = image.to(self.device), target.to(self.device)
        with torch.cuda.amp.autocast():
            out = model(image)
            loss = self.criterion(out, target)
            # import IPython; IPython.embed(); exit(1)
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

        self.logger.log_metric("batch_train_loss", loss, step=step)
        return loss

    @torch.no_grad()
    def test_step(self, model, image, step):
        """Log original image and cropped image for test"""
        image = image.to(self.device)
        with torch.cuda.amp.autocast():
            out = self.model(image)
        grid_image = torchvision.utils.make_grid(image.cpu(), nrow=4)
        grid_out = torchvision.utils.make_grid(out.detach().cpu(), nrow=4)
        self.logger.log_image(grid_image.permute(1,2,0), step=step, name="Orig")
        self.logger.log_image(grid_out.permute(1,2,0), step=step, name="SoftMask")

        # result = out.clamp(0,1).round().expand_as(image) * image
        result = out.expand_as(image)*image
        grid_result = torchvision.utils.make_grid(result.detach().cpu(), nrow=4)
        self.logger.log_image(grid_result.permute(1,2,0), step=step, name="Result")
        # import IPython; IPython.embed(); exit(1)
if __name__=="__main__":
    logger = comet_ml.Experiment(
        api_key=args.api_key,
        project_name="auto-crop"
    )
    experiment_name = get_experiment_name(args)
    logger.set_name(experiment_name)
    train_dl, test_dl = get_dataloader(args)
    trainer = Trainer(args, logger)
    trainer.fit(train_dl, test_dl)
    filename=f"weights/{args.model_name}_last.pth"
    torch.save(trainer.model.state_dict(), f=filename)
    logger.log_asset(filename, filename.split("/")[-1])