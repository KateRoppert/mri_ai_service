import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

IN = "/home/ubuntu/mri_ai_service/demo_workspace/input/1_06_1921/nifti/sub-001/ses-001/anat/sub-001_ses-001_t1.nii.gz"      # путь к одному T1

img  = sitk.ReadImage(IN, sitk.sitkFloat32)
mask = sitk.OtsuThreshold(img, 0, 1, 200)

shrink = 4
img_s  = sitk.Shrink(img,  [shrink]*img.GetDimension())
mask_s = sitk.Shrink(mask, [shrink]*mask.GetDimension())
corr = sitk.N4BiasFieldCorrectionImageFilter()
corr.SetMaximumNumberOfIterations([50, 50, 50, 50])
corr.Execute(img_s, mask_s)

log_bias  = corr.GetLogBiasFieldAsImage(img)   # поле в полном разрешении
bias      = sitk.Exp(log_bias)
corrected = img / bias

A = sitk.GetArrayFromImage(img)        # оси sitk: (z, y, x)
B = sitk.GetArrayFromImage(bias)
C = sitk.GetArrayFromImage(corrected)
M = sitk.GetArrayFromImage(mask) > 0

z = A.shape[0] // 2                     # средний аксиальный срез
a, b, c, m = A[z], B[z], C[z], M[z]
b_masked = np.ma.masked_where(~m, b)    # поле показываем только внутри мозга

fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
ax[0].imshow(np.rot90(a), cmap="gray"); ax[0].set_title("Original")
ax[1].imshow(np.rot90(a), cmap="gray")
im = ax[1].imshow(np.rot90(b_masked), cmap="jet", alpha=0.6)
ax[1].set_title("Estimated bias field"); fig.colorbar(im, ax=ax[1], fraction=0.046)
ax[2].imshow(np.rot90(c), cmap="gray"); ax[2].set_title("After N4")
for a_ in ax: a_.axis("off")
plt.tight_layout()
plt.savefig("bias_field_figure.png", dpi=200, bbox_inches="tight")
print("saved bias_field_figure.png")