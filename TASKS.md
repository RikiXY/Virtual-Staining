# TASKS

## Now
- [X] Update README to the latest project version
- [ ] Re-check the full pipeline after recent refactoring
- [ ] Re-check path handling across preprocessing, training, test and evaluation
- [ ] Re-check logs, checkpoints and run metadata
- [ ] Re-check dataset and run folder assumptions
- [ ] Re-check evaluation and comparison tools end to end

## Next
- [ ] Add an SSIM loss term and observe training behavior
- [ ] Compare different loss weight settings
- [ ] Add data augmentation without geometric distortions
- [ ] Save augmentation settings in run metadata
- [ ] Improve qualitative result comparison across runs

## Later
- [ ] Explore integration with the autofluorescence study
- [ ] Test UNet3+ as generator 
- [ ] Test UNet++ as generator 
- [ ] Study a stronger discriminator 
- [ ] Test better perceptual and structural loss functions
- [ ] Add more informative evaluation metrics