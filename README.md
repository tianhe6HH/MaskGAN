
## MaskGAN

datasets
    |
    -<your dataset>
        |
        -trainA
        -trainB
        -testA
        -testB
Yolov8
    |
    -<your best>.pt

##### The Key Point: 
Searching "mask" in models/cycle_gan_model.py

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
