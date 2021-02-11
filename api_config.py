class Config:
    def __init__(self):
        self.cuda = True
        self.arch = "resnet"
        self.pretrained_model = "./results/pretrained/resnet50-19c8e357.pth"
        self.model = "results/run-1/models/final.pth"
