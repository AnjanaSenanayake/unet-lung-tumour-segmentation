import json

class Config:    
    def __init__(self, file_path):
        try:
            with open(file_path, "rb") as jsonfile:      
                configs = json.load(jsonfile)
                
                self.num_epochs = configs['epochs']
                self.train_size = configs['train_size']
                self.batch_size = configs['batch_size']
                self.search_size = configs['search_size']
                self.template_size = configs['template_size']
                self.val_ratio = configs['val_ratio']
                self.train_num_workers = configs['train_num_workers']
                self.val_num_workers = configs['val_num_workers']
                self.weight_decay = configs['weight_decay']
                self.momentum = configs['momentum']
                self.gradient_clipping = configs['gradient_clipping']
                self.gradient_scaler = configs['gradient_scaler']
                self.autocast = configs['autocast']
                self.lr = configs['lr']
                self.lr_min = configs['lr_min']
                self.lr_schedular_mode = configs['lr_schedular_mode']
                self.lr_schedular_patience = configs['lr_schedular_patience']
                self.lr_schedular_factor = configs['lr_schedular_factor']
                self.lr_schedular_decay_rate = configs['lr_schedular_decay_rate']
        except FileNotFoundError as e:
            print(f"{e}: Make sure to have it in same directory as in trainer.py")
            exit(1)
    
    def toJSON(self):
        return {
            'epochs': self.num_epochs,
            'train_size': self.train_size,
            'batch_size': self.batch_size,
            'search_size': self.search_size,
            'template_size': self.template_size,
            'val_ratio': self.val_ratio,
            'train_num_workers': self.train_num_workers,
            'val_num_workers': self.val_num_workers,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'gradient_clipping': self.gradient_clipping,
            'gradient_scaler': self.gradient_scaler,
            'autocast': self.autocast,
            'lr': self.lr,
            'lr_min': self.lr_min,
            'lr_schedular_mode': self.lr_schedular_mode,
            'lr_schedular_patience': self.lr_schedular_patience,
            'lr_schedular_factor': self.lr_schedular_factor,
            'lr_schedular_decay_rate': self.lr_schedular_decay_rate
        }