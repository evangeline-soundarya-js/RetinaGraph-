import cv2
import numpy as np

class FundusValidator:
    def __init__(self):
        # Configuration for plausible fundus characteristics
        self.min_dim = 100
        self.max_dim = 4000
        self.min_mean_intensity = 5.0
        self.max_mean_intensity = 250.0
        self.min_std_intensity = 5.0

    def validate(self, img_np: np.ndarray):
        """
        Performs heuristic plausibility checks on the image array.
        Returns a dictionary with status, reason, and individual checks.
        """
        checks = {
            "decodable": True, # If we got here, it's decoded
            "dimensions_valid": False,
            "brightness_valid": False,
            "color_plausible": False,
            "structure_plausible": False
        }
        
        reason = []

        # 1. Dimensions check
        h, w = img_np.shape[:2]
        if self.min_dim <= h <= self.max_dim and self.min_dim <= w <= self.max_dim:
            checks["dimensions_valid"] = True
        else:
            reason.append(f"Image dimensions ({w}x{h}) are outside expected bounds.")

        # 2. Brightness / Contrast check
        mean_intensity = np.mean(img_np)
        std_intensity = np.std(img_np)
        
        if mean_intensity < self.min_mean_intensity:
            reason.append("Image is too dark (almost completely black).")
        elif mean_intensity > self.max_mean_intensity:
            reason.append("Image is too bright (almost completely white).")
        elif std_intensity < self.min_std_intensity:
            reason.append("Image is too uniform (blank or low contrast).")
        else:
            checks["brightness_valid"] = True

        # 3. Color Plausibility (Fundus images are usually reddish/orange)
        # Check if R channel is generally dominant over B channel.
        # This is a heuristic and may fail on some abnormal images, but good for filtering out portraits/landscapes.
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # OpenCV might be RGB or BGR depending on how it was loaded. 
            # In image_processor, we convert PIL (RGB) to numpy, so it's RGB.
            R, G, B = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
            mean_R = np.mean(R)
            mean_G = np.mean(G)
            mean_B = np.mean(B)
            
            # Typical fundus: R > G > B or R is dominant. 
            # Let's be lenient: R should be > B.
            if mean_R > mean_B:
                checks["color_plausible"] = True
            else:
                reason.append("Color profile does not match typical retinal photography (Red channel is not dominant).")
        else:
            # If it's grayscale, we skip color check but mark it true to pass
            checks["color_plausible"] = True

        # 4. Structural Plausibility
        # A simple check: most fundus images have a circular field of view, so the corners are dark.
        # Let's check the intensity of the 4 corners vs the center.
        cx, cy = w // 2, h // 2
        r = min(w, h) // 4
        center_patch = img_np[cy-r:cy+r, cx-r:cx+r]
        
        corner_r = min(w, h) // 10
        top_left = img_np[0:corner_r, 0:corner_r]
        top_right = img_np[0:corner_r, w-corner_r:w]
        bottom_left = img_np[h-corner_r:h, 0:corner_r]
        bottom_right = img_np[h-corner_r:h, w-corner_r:w]
        
        mean_center = np.mean(center_patch)
        mean_corners = np.mean([np.mean(top_left), np.mean(top_right), np.mean(bottom_left), np.mean(bottom_right)])
        
        # If center is significantly brighter than corners, it's structurally plausible as a circular FOV.
        # Or if it's tightly cropped, this might fail, so we make it lenient.
        if mean_center > mean_corners + 5.0 or checks["color_plausible"]:
            # If it passes color or circularity, we mark structure plausible
            checks["structure_plausible"] = True
        else:
            reason.append("Image lacks expected retinal field structure (e.g. circular mask).")

        # Compile final status
        all_passed = all(checks.values())
        status = "accepted" if all_passed else "rejected"
        final_reason = "Image is suitable for analysis." if all_passed else " | ".join(reason)

        return {
            "status": status,
            "reason": final_reason,
            "checks": checks
        }
