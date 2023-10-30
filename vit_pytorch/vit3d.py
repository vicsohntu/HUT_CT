import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        #import pdb; pdb.set_trace()
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            y, at = attn(x)
            x = y + x
            x = ff(x) + x
        return x, at

class ViT3d(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        self.heads=heads
        self.num_patches_h = (image_height // patch_height) 
        self.num_patches_w =(image_width // patch_width) 
        self.num_patches_d =(image_depth // patch_depth)
        num_patches =self.num_patches_h* self.num_patches_w* self.num_patches_d
        patch_dim = channels * patch_height * patch_width * patch_depth
        self.patch_dim=patch_dim
        self.channels=channels
        self.ph=patch_height
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # ( 1, 4608, 512)
        # 512 - > 1024
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1 = patch_height, p2 = patch_width, p3 = patch_depth),
            nn.Linear(patch_dim, dim),
        )
##        self.to_convpatch_embedding = nn.Sequential(
##          nn.Conv3d(1, dim//4, 3,2,1,1),
##          nn.Conv3d(dim//4, dim//2, 3,2,1,1),
##          nn.Conv3d(dim//2, dim, 3,2,1,1),
##          Rearrange('b c h w d -> b (h w d) c'),
##        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.hidden_dim=2048
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.layer1 = nn.Sequential(nn.Linear(in_features=dim, out_features= self.hidden_dim, bias=False ),
                                    #nn.BatchNorm1d(self.hidden_dim),
                                    nn.ReLU(inplace=True)
                                   )

        self.layer2 = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False),
                                    #nn.BatchNorm1d(self.hidden_dim),
                                    nn.ReLU(inplace=True)
                                   )
        
        self.layer3 = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=num_classes, bias=False),
                                    #nn.BatchNorm1d(num_classes)
                                   )

    def forward(self, img):
        #import pdb; pdb.set_trace()
        xxx = self.to_patch_embedding(img) # Linear
        #x = self.to_convpatch_embedding(img) # Conv type
        #import pdb; pdb.set_trace()
        b, num, _ = xxx.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, xxx), dim=1)
        #import pdb; pdb.set_trace()
        x += self.pos_embedding[:, :(num + 1)]
        x = self.dropout(x)

        x, attn1 = self.transformer(x)

        x1 = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #x3 = self.mlp_head(x2)
        #import pdb; pdb.set_trace()
        x3 = self.layer1(x1)
        
        x4 = self.layer2(x3)
        x4a = self.to_latent(x4)        
        x5 = self.layer3(x4a)
        #import pdb; pdb.set_trace()
        attn1=attn1[:, :, 0, 1:].reshape(b, self.heads, self.num_patches_h, self.num_patches_w, self.num_patches_d)
        return attn1, x5

class ViT3dpatch(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # ( 1, 4608, 512)
        # 512 - > 1024
        self.to_convpatch_embedding = nn.Sequential(
          nn.Conv3d(1, dim//2, 3,2,1,1),
          #nn.Conv3d(dim//4, dim//2, 3,2,1,1),
          nn.Conv3d(dim//2, dim, 3,2,1,1),
          Rearrange('b c h w d -> b (h w d) c'),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.hidden_dim=2048
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.combiner = nn.Sequential(
            nn.ConvTranspose3d(4,4,3,2,1,1),
            nn.ConvTranspose3d(4,2,3,2,1,1),
        )
        
        self.layer1 = nn.Sequential(nn.Linear(in_features=dim, out_features= self.hidden_dim, bias=False ),
                                    #nn.BatchNorm1d(self.hidden_dim),
                                    nn.ReLU(inplace=True)
                                   )

        self.layer2 = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False),
                                    #nn.BatchNorm1d(self.hidden_dim),
                                    nn.ReLU(inplace=True)
                                   )
        
        self.layer3 = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=num_classes, bias=False),
                                    #nn.BatchNorm1d(num_classes)
                                   )

    def forward(self, img):
        x = self.to_convpatch_embedding(img) # Conv type
        #import pdb; pdb.set_trace()
        b, num, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        #import pdb; pdb.set_trace()
        x += self.pos_embedding[:, :(num + 1)]
        x = self.dropout(x)
        x, at = self.transformer(x)
        x1 = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #x3 = self.mlp_head(x2)
        #import pdb; pdb.set_trace()
        x3 = self.layer1(x1)
        x4 = self.layer2(x3)
        x4a = self.to_latent(x4)        
        x5 = self.layer3(x4a)
        att=at[:,:,0,1:].reshape(b,4,10,10,12)
        att=self.combiner(att)
        #import pdb; pdb.set_trace()
        return x5, att


class ViT3dpatchH(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth
        self.to_convpatch_embedding0 = nn.Sequential(
            nn.Conv3d(1, dim//2, 3,2,1,1),
        )
          #nn.Conv3d(dim//4, dim//2, 3,2,1,1),
        self.to_convpatch_embedding1 = nn.Sequential(
            nn.Conv3d(dim//2, dim, 3,2,1,1),
            Rearrange('b c h w d -> b (h w d) c'),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.hidden_dim=2048
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.combiner0 = nn.Sequential(
            nn.ConvTranspose3d(4,256,3,2,1,1),
            nn.InstanceNorm3d(256),
            nn.Dropout(0.1),
            #nn.ReLU(),
        )
        self.combiner1 = nn.Sequential(
            nn.ConvTranspose3d(512,2,3,2,1,1),
            nn.Dropout(0.1),
            #nn.ReLU(),
        )
    def forward(self, img):
        x0 = self.to_convpatch_embedding0(img) # Conv type
        x = self.to_convpatch_embedding1(x0) # Conv type
        #import pdb; pdb.set_trace()
        b, num, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        #import pdb; pdb.set_trace()
        x += self.pos_embedding[:, :(num + 1)]
        x = self.dropout(x)
        _, at = self.transformer(x)
        att=at[:,:,0,1:].reshape(b,4,10,10,12)
        att0=self.combiner0(att)
        att1=torch.cat((att0, x0), dim=1)
        att=self.combiner1(att1)
        #import pdb; pdb.set_trace()
        return att

#full 4-> 40,40,48, patch 4 -> 10,10,12, patch 2 -> 5, 5, 6 
class ViT3dpatch4(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth
        self.to_convpatch_embedding0 = nn.Sequential(
            nn.Conv3d(1, dim//2, 3,2,1,1),
        )
          #nn.Conv3d(dim//4, dim//2, 3,2,1,1),
        self.to_convpatch_embedding1 = nn.Sequential(
            nn.Conv3d(dim//2, dim, 3,2,1,1),
            Rearrange('b c h w d -> b (h w d) c'),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.hidden_dim=2048
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.combiner0 = nn.Sequential(
            nn.ConvTranspose3d(4,256,3,2,1,1),
            nn.InstanceNorm3d(256),
            nn.Dropout(0.1),
            #nn.ReLU(),
        )
        self.combiner1 = nn.Sequential(
            nn.ConvTranspose3d(512,2,3,2,1,1),
            nn.Dropout(0.1),
            #nn.ReLU(),
        )
    def forward(self, img):
        x0 = self.to_convpatch_embedding0(img) # Conv type
        x = self.to_convpatch_embedding1(x0) # Conv type
        #import pdb; pdb.set_trace()
        b, num, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        #import pdb; pdb.set_trace()
        x += self.pos_embedding[:, :(num + 1)]
        x = self.dropout(x)
        _, at = self.transformer(x)
        att=at[:,:,0,1:].reshape(b,4,10,10,12)
        att0=self.combiner0(att)
        att1=torch.cat((att0, x0), dim=1)
        att=self.combiner1(att1)
        #import pdb; pdb.set_trace()
        return att

class ViT3dBYOL(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # ( 1, 4608, 512)
        # 512 - > 1024
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1 = patch_height, p2 = patch_width, p3 = patch_depth),
            nn.Linear(patch_dim, dim),
        )
##        self.to_convpatch_embedding = nn.Sequential(
##          nn.Conv3d(4, dim//4, 3,2,1,1),
##          nn.Conv3d(dim//4, dim//2, 3,2,1,1),
##          nn.Conv3d(dim//2, dim, 3,2,1,1),
##          Rearrange('b c h w d -> b (h w d) c'),
##        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.hidden_dim=2048
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        #import pdb; pdb.set_trace()
        x = self.to_patch_embedding(img) # Linear
        #x = self.to_convpatch_embedding(img) # Conv type
        #import pdb; pdb.set_trace()
        b, num, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        #import pdb; pdb.set_trace()
        x += self.pos_embedding[:, :(num + 1)]
        x = self.dropout(x)

        x, at = self.transformer(x)

        x1 = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x2 = self.to_latent(x1)     
        x3 = self.mlp_head(x2)
        #import pdb; pdb.set_trace()
       
        return x3, at

class ViT3dpatch20x20x24(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth
        self.heads=heads
        self.px=image_size[0]//patch_size
        self.py=image_size[1]//patch_size
        self.pz=image_size[2]//patch_size
        self.to_convpatch_embedding0 = nn.Sequential(
            nn.Conv3d(1, dim//2, 3,2,1,1),
        )
          #nn.Conv3d(dim//4, dim//2, 3,2,1,1),
        self.to_convpatch_embedding1 = nn.Sequential(
            nn.Conv3d(dim//2, dim, 3,2,1,1),
            Rearrange('b c h w d -> b (h w d) c'),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.hidden_dim=2048
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.combiner0 = nn.Sequential(
            nn.ConvTranspose3d(self.heads,256,3,2,1,1),
            nn.InstanceNorm3d(256),
            nn.Dropout(0.1),
            #nn.ReLU(),
        )
        self.combiner1 = nn.Sequential(
            nn.ConvTranspose3d(512,2,3,2,1,1),
            nn.Dropout(0.1),
            #nn.ReLU(),
        )
    def forward(self, img):
        x0 = self.to_convpatch_embedding0(img) # Conv type
        x = self.to_convpatch_embedding1(x0) # Conv type
        #import pdb; pdb.set_trace()
        b, num, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        #import pdb; pdb.set_trace()
        x += self.pos_embedding[:, :(num + 1)]
        x = self.dropout(x)
        _, at = self.transformer(x)
        att=at[:,:,0,1:].reshape(b, self.heads, self.px,self.py,self.pz)
        att0=self.combiner0(att)
        att1=torch.cat((att0, x0), dim=1)
        att=self.combiner1(att1)
        #import pdb; pdb.set_trace()
        return att

def get_module_device(module):
    return next(module.parameters()).device

class ViTDiced3d(nn.Module):
    def __init__(self, *, org_size, image_size, patch_size, num_classes=10, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.,device):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth
        self.num_x=org_size[0]//image_size[0]
        self.num_y=org_size[1]//image_size[1]
        self.num_z=org_size[2]//image_size[2]
        self.dicex=image_size[0]
        self.dicey=image_size[1]
        self.dicez=image_size[2]
        self.img_patch=list()
        self.patch_encoder=nn.ModuleList()
        for i in range(self.num_x*self.num_y*self.num_z):
            self.patch_encoder.append(ViT3dpatch20x20x24(
                channels=channels,
                image_size = image_size,
                patch_size = patch_size,
                num_classes = num_classes,
                dim = dim,
                depth = depth,
                heads = heads,
                mlp_dim = mlp_dim,
                dropout=dropout,
                emb_dropout=emb_dropout
            ).to(device))
        #import pdb; pdb.set_trace()
        self.to(device)            
        self.outp=torch.randn(1, 2, org_size[0], org_size[1], org_size[2], device=device)
        self.forward(torch.randn(1,1, org_size[0], org_size[1], org_size[2], device=device))
    def forward(self, img):
        for k in range(self.num_z):
            for j in range(self.num_y):
                for i in range(self.num_x):
                    px=i*self.dicex; py=j*self.dicey; pz=k*self.dicez
                    self.img_patch.append(self.patch_encoder[i+j*self.num_y+k*self.num_x*self.num_y](img[:, :, px:px+self.dicex, py:py+self.dicey, pz:pz+self.dicez]))
        #import pdb; pdb.set_trace()
        for k in range(self.num_z):
            for j in range(self.num_y):
                for i in range(self.num_x):
                    px=i*self.dicex; py=j*self.dicey; pz=k*self.dicez
                    self.outp[0, :, px:px+self.dicex, py:py+self.dicey, pz:pz+self.dicez] = self.img_patch[i+j*self.num_y+k*self.num_x*self.num_y]
        return self.outp

class ViTDicedCube(nn.Module):
    def __init__(self, *, org_size, image_size, patch_size, num_classes=10, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.,device):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth
        self.num_x=org_size[0]//image_size[0]
        self.num_y=org_size[1]//image_size[1]
        self.num_z=org_size[2]//image_size[2]
        self.dicex=image_size[0]
        self.dicey=image_size[1]
        self.dicez=image_size[2]
        self.img_patch=list()
        self.patch_encoder=nn.ModuleList()
        for i in range(self.num_x*self.num_y*self.num_z):
            self.patch_encoder.append(ViT3dpatch20x20x24(
                channels=channels,
                image_size = image_size,
                patch_size = patch_size,
                num_classes = num_classes,
                dim = dim,
                depth = depth,
                heads = heads,
                mlp_dim = mlp_dim,
                dropout=dropout,
                emb_dropout=emb_dropout
            ).to(device))
        #import pdb; pdb.set_trace()
        self.to(device)            
        self.outp=torch.randn(1, 2, org_size[0], org_size[1], org_size[2], device=device)
        self.forward(torch.randn(1,1, org_size[0], org_size[1], org_size[2], device=device))
    def forward(self, img):
        for k in range(self.num_z):
            for j in range(self.num_y):
                for i in range(self.num_x):
                    px=i*self.dicex; py=j*self.dicey; pz=k*self.dicez
                    self.img_patch.append(self.patch_encoder[i+j*self.num_y+k*self.num_x*self.num_y](img[:, :, px:px+self.dicex, py:py+self.dicey, pz:pz+self.dicez]))
        #import pdb; pdb.set_trace()
        for k in range(self.num_z):
            for j in range(self.num_y):
                for i in range(self.num_x):
                    px=i*self.dicex; py=j*self.dicey; pz=k*self.dicez
                    self.outp[0, :, px:px+self.dicex, py:py+self.dicey, pz:pz+self.dicez] = self.img_patch[i+j*self.num_y+k*self.num_x*self.num_y]
        return self.outp
