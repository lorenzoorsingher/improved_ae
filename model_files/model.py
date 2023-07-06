import torch.nn as nn


class SimplerAE2(nn.Module):
    def __init__(self):
        super(SimplerAE2, self).__init__()

        #([(W - K + 2P)/S] + 1)

        #input img [BS, 1, 80,80]

        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=3,              
                stride=2,                   
                padding=0,                  
            ),                              
            nn.ReLU(),
            nn.BatchNorm2d(16)                      
        )# out [BS, 16, 39, 39]

        # in [BS, 16, 39, 39]
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=3,              
                stride=2,                   
                padding=0                  
            ),     
            nn.ReLU(),
            nn.BatchNorm2d(32)                        
        )# out [BS, 32, 19, 19]
        
        # in [BS, 32, 19, 19]
        self.flat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(11552, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 11552),
            nn.ReLU()
        )# out [BS, 11552]

        #output_size = strides * (input_size-1) + kernel_size - 2*padding
        
        # in [BS, 32, 19, 19]
        self.trans1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16)  
        )# out [BS, 16, 39, 39]

        # in [BS, 16, 39, 39]
        self.trans2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                stride=2,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(3)  
        )# out [BS, 3, 79, 79]

        #self.sig = nn.Sigmoid()
        #self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = x.view(x.size(0), 32, 19, 19)
        x = self.trans1(x) 
        output = self.trans2(x)
        #output = self.tanh(x)
         
        #output = self.sig(x)
        return output, x    # return x for visualization
