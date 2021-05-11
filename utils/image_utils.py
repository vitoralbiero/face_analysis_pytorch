import numpy as np
import scipy.ndimage


def merge_bisenet_annotations(mask):
    # remove ears and earrings
    mask[np.logical_and(mask >= 7, mask <= 9)] = 0
    # merge face parts
    mask[np.logical_and(mask > 1, mask <= 13)] = 1
    # remove hair, background, hat, neck, etc
    mask[mask > 13] = 0

    return mask


# This is used to calculate the blur amount based on a polynomial
def get_curve(max_blur, val, order=4):
    val = (1 / (max_blur ** (order - 1))) * (val ** order)
    return val


def blur_image(img, mask, max_blur=10, curved=True):
    mask = merge_bisenet_annotations(mask)  # merge annotations
    blurred_img = img.copy()  # to hold final image
    mask_max = max(mask.max(), 1)  # Make sure there are no 0 maxes
    mask = mask * (255.0 / mask_max)  # transform to 0 to 255 range
    mask = scipy.ndimage.gaussian_filter(
        mask, 5
    )  # Blur this image so no hard boundaries between different levels
    mask_max = max(mask.max(), 1)
    scaled_img = mask / (mask_max / max_blur)
    # Scale image back to range 0 to sigma max so we get correct blur level

    scaled_img = np.around(scaled_img, decimals=1)  # Round to one decimal place
    uniq_vals = np.unique(scaled_img)
    for scaled_val in uniq_vals:  # for all possible blur levels
        blur_amount = round(max_blur - scaled_val, 1)
        if curved:  # based on polynomial instead of linear
            blur_amount = get_curve(max_blur, blur_amount, order=4)

        if blur_amount != 0:  # if blurring required
            indices = np.where(
                scaled_img == scaled_val
            )  # where this level of blur should be applied

            if img.ndim == 2:
                qwerty = scipy.ndimage.gaussian_filter(img, blur_amount)
            else:
                qwerty = img.copy()
                for i in range(qwerty.shape[2]):
                    qwerty[..., i] = scipy.ndimage.gaussian_filter(
                        img[..., i], blur_amount
                    )

            blurred_img[indices] = qwerty[
                indices
            ]  # apply that blur level to the specified pixels

    return blurred_img
