
## MaskGAN

datasets
    |
    -<your dataset>
        |
        -trainA
        -trainB
        -testA
        -testB

##### The Key Point: 
Searching "mask" in models/mask_gan_model.py

##### Yolo-seg(not key)
You need to train your yolo pth files.

###### Explanation:
###### Generators:
            G_A: A -> B; G_B: B -> A.
            A->fake B-> fake B Mask, A->real A Mask, caculate loss of (fake B Mask) and (real A Mask)
            B->fake A-> fake A Mask, B->real B Mask, caculate loss of (fake A Mask) and (real B Mask)

###### Discriminators:
            D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.

### Train Example:
    Train a MaskGAN model:
        python train.py --dataroot ./datasets/PAM_HE_kidney --name PAM_HE_kidney1 --model mask_gan

### Test Example:
        python test.py --dataroot datasets/PAM_HE_kidney/testB --name PAM_HE_kidney1 --model test --no_dropout


